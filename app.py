import os
import re
import json
import time
import logging
from typing import Annotated, Optional
from datetime import datetime, timedelta
from typing_extensions import TypedDict

# Langchain & LangGraph Imports
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Import the client from the separate file
try:
    from nexhealth_client import NexHealthClient
except ImportError:
    logging.error("`nexhealth_client.py` not found. Please ensure it's in the same directory.")
    exit()

# ==============================================================================
# 1. Setup & Configuration
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

try:
    nex_client = NexHealthClient()
except Exception as e:
    logging.error(f"Could not initialize NexHealth Client: {e}")
    logging.error("Please check your .env file and API credentials.")
    nex_client = None

# Mappings for location 
SAMPLE_LOCATION_MAP = {
    331668: "Green River Dental"
}


# ==============================================================================
# 2. Tool Definitions
# ==============================================================================

@tool
def validate_personal_info(name: str, phone_number: str, dob: str) -> str:
    """
    Validates the patient's name, 10-digit phone number, and date of birth (YYYY-MM-DD format).
    This is a required first step before searching for a patient.
    """
    logging.info(f"--- Calling validate_personal_info for {name} ---")
    errors = []
    if not name or len(name.split()) < 2:
        errors.append("Invalid name. Please provide a full name.")
    if not re.match(r"^\d{10}$", phone_number):
        errors.append("Invalid phone number. Please provide 10 digits only, e.g., 5551234567.")
    try:
        datetime.strptime(dob, "%Y-%m-%d")
    except ValueError:
        errors.append("Invalid date of birth. Please use YYYY-MM-DD format, e.g., 1990-05-15.")
    if errors:
        return json.dumps({"is_valid": False, "errors": errors})
    return json.dumps({"is_valid": True, "errors": []})



@tool
def search_patient(name: str, phone_number: str, dob: str) -> str:
    """
    Searches for an existing patient record using their validated personal info.
    Returns 'no_patient_found' or a JSON object with patient details.
    """
    logging.info(f"--- Calling search_patient for {name} ---")
    if not nex_client:
        return json.dumps({"status": "error", "message": "NexHealth client is not initialized."})
    

    search_result = nex_client.search_patients(name, phone_number, dob)
    logging.info(f"Search result: {search_result}")

    if isinstance(search_result, str):
        if search_result == "no_patient_found":
            return json.dumps({"status": "no_patient_found"})
        return json.dumps({"status": "error", "message": search_result})

    try:
        patient_data = search_result[0]

        if patient_data.get('status') == 'phone_number_mismatch':
            return json.dumps({"status": "phone_number_mismatch", "name": patient_data.get('name')})

        if patient_data.get('status') == 'verified_patient':
            result = {
                "status": "verified_patient",
                "patient_id": patient_data.get('id'),
                "name": patient_data.get('name'),
                "location_name": None,
                "provider_name": None 
            }

            upcoming_appts = patient_data.get('upcoming_appts', [])
            if upcoming_appts:
                first_appt = upcoming_appts[0]
                loc_id = first_appt.get('location_id')
                result["location_name"] = SAMPLE_LOCATION_MAP.get(loc_id, f"Location ID {loc_id}")
       
                result["provider_name"] = first_appt.get('provider_name')

            elif patient_data.get('location_ids'):
                first_loc_id = patient_data.get('location_ids')[0]
                result["location_name"] = SAMPLE_LOCATION_MAP.get(first_loc_id, f"Location ID {first_loc_id}")



            return json.dumps(result)

    except (IndexError, KeyError) as e:
        logging.error(f"Error parsing patient data: {e}")
        return json.dumps({"status": "error", "message": "Could not parse patient data from API."})

    return json.dumps({"status": "no_patient_found"})


@tool
def get_available_slots(location: str, provider: str) -> str:
    """
    Returns a list of sample appointment slots for a specific location and provider.
    Call this ONLY after you have confirmed the location and provider with the user.
    """
    logging.info(f"--- Calling get_available_slots for Location: {location}, Provider: {provider} ---")
    slots = {
        "date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        "available_slots": ["10:00 AM", "10:30 AM", "11:00 AM", "02:00 PM", "02:30 PM"]
    }
    return json.dumps(slots)

@tool
def book_appointment(patient_name: str, location: str, provider: str, slot_start_time: str) -> str:
    """
    Books the appointment using the patient's name and the selected location, provider, and time.
    Call this as the final step once all information has been gathered and confirmed.
    """
    appointment_details = f"Location: {location}, Provider: {provider}"
    logging.info(f"--- BOOKING for {patient_name} at {appointment_details} for {slot_start_time} ---")
    confirmation = {
        "status": "confirmed",
        "confirmation_id": f"CONF_{int(time.time())}",
        "patient_name": patient_name,
        "details": appointment_details,
        "appointment_time": slot_start_time
    }
    return json.dumps(confirmation)

tools = [
    validate_personal_info,
    search_patient,
    get_available_slots,
    book_appointment
]

# ==============================================================================
# 3. Agent & Graph State Definition
# ==============================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]



SYSTEM_PROMPT = """
You are a specialized medical appointment booking assistant. Your goal is to book an appointment by following a strict, non-deviating workflow.

**Step 1: Collect & Validate Patient Info**
1. Greet the user and ask for their full name, 10-digit phone number, and date of birth (YYYY-MM-DD).
2. Once you have all three pieces of information, you MUST call the `validate_personal_info` tool.
3. If validation fails, state the exact errors and ask the user to correct them. Do not proceed.
4. Once validation is successful, you MUST immediately call the `search_patient` tool.

**Step 2: Handle Patient Search Results (Workflow Branching)**
This is a critical step. You will follow one of two paths based on the tool's output.

**PATH A: Existing Patient (`"status": "verified_patient"`)**
The tool will return the patient's name, a location, and possibly a provider.
1.  **Confirm Details:** Address the patient by name and confirm the details found. For example: "Thanks, John. I found your record. I see you're with Green River Dental. Is that correct?"
2.  **Handle Provider Info:**
    * **If a provider_name was returned:** Confirm that too. "And you usually see Dr. Smith. Would you like to book with them again?"
    * **If provider_name was NOT returned (is null):** The patient exists but we don't know their preferred provider. YOU MUST ASK THEM. Say: "Is there a specific provider you'd like to see at Green River Dental?"
3.  **Get Slots:** Once location and provider are confirmed (or newly provided), call `get_available_slots` with that specific location and provider.
4.  **Book:** After the user chooses a slot, call `book_appointment` with all the confirmed details.

**PATH B: New Patient (`"status": "no_patient_found"`)**
1.  **Acknowledge:** Inform the user you couldn't find a record and will create one for them.
2.  **Collect Location:** Your IMMEDIATE next question MUST be to ask for their preferred clinic location. Do not ask about anything else.
3.  **Collect Provider:** AFTER they provide a location, your NEXT question MUST be to ask for their preferred provider at that location.
4.  **Get Slots:** ONLY after you have gathered their name, phone, DOB, location, and provider, call `get_available_slots` with the provided location and provider.
5.  **Book:** After the user chooses a slot, call `book_appointment` with all the gathered details.

**General Rules:**
- Do not deviate from these workflows.
- Do not ask for information you already have. Use the information returned by the tools.
- Be polite and efficient.
"""

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def chatbot_agent(state: State):
    logging.info("--- Calling Agent ---")
    messages_with_system_prompt = [("system", SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages_with_system_prompt)
    return {"messages": [response]}

# ==============================================================================
# 4. Graph Construction & Execution
# ==============================================================================

builder = StateGraph(State)
builder.add_node("agent", chatbot_agent)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "tools", END: END}
)
builder.add_edge("tools", "agent")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

logging.info("Graph compiled successfully!")
logging.info("Ready to chat.")

def run_conversation(user_input, thread_config):
    if user_input.lower() == 'exit':
        return "Conversation ended."
    logging.info(f"USER: {user_input}")
    final_state = graph.invoke(
        {"messages": [("user", user_input)]},
        config=thread_config
    )
    last_message = final_state["messages"][-1]
    if hasattr(last_message, 'content') and last_message.content:
        response_content = last_message.content
        logging.info(f"ASSISTANT: {response_content}")
        return response_content
    return "The conversation has ended."

if __name__ == "__main__":
    config = {"configurable": {"thread_id": f"appointment_thread_{int(time.time())}"}}
    logging.info("--- Starting New Appointment Conversation ---")
    logging.info("Type 'exit' to end.")
    run_conversation("Hi, I'd like to book an appointment.", config)
    while True:
        try:
            user_msg = input("USER: ")
            if user_msg.lower() == 'exit':
                logging.info("ASSISTANT: Goodbye!")
                break
            run_conversation(user_msg, config)
        except KeyboardInterrupt:
            logging.info("\nASSISTANT: Conversation ended by user.")
            break
        except Exception as e:
            logging.error(f"An unhandled error occurred: {e}")
            break