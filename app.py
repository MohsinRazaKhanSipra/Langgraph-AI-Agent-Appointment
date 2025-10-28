import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from nexhealth_client import NexHealthClient  
from pydantic import BaseModel, Field


load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0) 


class AppointmentState(BaseModel):
    """The complete state for the appointment booking process."""
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Patient Info
    name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None
    dob: Optional[str] = None 
    
    # Patient Status
    patient_id: Optional[str] = None
    is_existing_patient: Optional[bool] = None
    upcoming_appt_id: Optional[int] = None

    # Appointment Details
    location: Optional[int] = None
    location_name: Optional[str] = None
    provider: Optional[int] = None
    provider_name: Optional[str] = None
    operatory_id: Optional[int] = None
    slot_time: Optional[str] = None
    appointment_type_id: int = 1  


    location_options: List[Dict] = Field(default_factory=list)
    provider_options: List[Dict] = Field(default_factory=list)
    slot_options: List[Dict] = Field(default_factory=list)



@tool
def search_patient(name: str, phone_number: str, dob: str, location_id: int = 331668) -> Union[str, List[dict]]:
    """
    Search for an existing patient by name, phone, dob, and location.
    This is called *after* collecting the user's basic info.
    """
    client = NexHealthClient()
    return client.search_patients(name, phone_number, dob, location_id)

@tool
def get_locations() -> List[dict]:
    """Fetch all available practice locations."""
    return NexHealthClient().get_locations()

@tool
def get_providers(location_id: int) -> List[dict]:
    """Get a list of providers (doctors) for a given location_id."""
    return NexHealthClient().get_providers(location_id)

@tool
def get_slots(location_ids: List[int], provider_ids: List[int], start_date: str, days: int) -> List[dict]:
    """
    Fetch available appointment slots.
    The LLM is responsible for calculating start_date and days.
    For example, if the user asks for "tomorrow", the LLM should calculate tomorrow's date.
    """
    client = NexHealthClient()
    return client.get_available_slots(start_date, days, location_ids, provider_ids)

@tool
def view_appointment(appointment_id: int, location_id: int) -> Union[str, dict]:
    """Retrieve details for a specific upcoming appointment by its ID."""
    client = NexHealthClient()
    return client.view_appointment(appointment_id, location_id)

@tool
def create_patient(
    provider_id: int,
    first_name: str,
    last_name: str,
    email: str,
    phone_number: str,
    date_of_birth: str,
    location_id: int
) -> Dict[str, Any]:
    """Create a new patient record. Call this *before* booking for new patients."""
    client = NexHealthClient()
    return client.create_patient(
        provider_id=provider_id,
        first_name=first_name,
        last_name=last_name,
        email=email,
        phone_number=phone_number,
        date_of_birth=date_of_birth,
        location_id=location_id
    )

@tool
def make_appointment(
    patient_id: str,
    location_id: int,
    provider_id: int,
    operatory_id: int,
    start_time: str,
    appointment_type_id: int
) -> dict:
    """Book the final appointment. Call this *only* after user confirmation."""
    client = NexHealthClient()
    return client.make_appointment(
        patient_id,
        location_id,
        provider_id,
        operatory_id,
        start_time,
        appointment_type_id
    )


@tool
def update_personal_info(
    name: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    phone_number: Optional[str] = None,
    dob: Optional[str] = None,
    email: Optional[str] = None
) -> dict:
    """
    Updates personal information in the state.
    Use this when the user provides info (e.g., "My name is John Doe").
    If a full name is given, split it into first_name and last_name.
    """
    updates = {}
    if name:
        updates["name"] = name
        if " " in name and not first_name and not last_name:
            parts = name.split(" ", 1)
            updates["first_name"] = parts[0]
            updates["last_name"] = parts[1] if len(parts) > 1 else ""
    if first_name: updates["first_name"] = first_name
    if last_name: updates["last_name"] = last_name
    if phone_number: updates["phone_number"] = phone_number
    if dob: updates["dob"] = dob
    if email: updates["email"] = email
    
    return {"status": "info updated", "updated_fields": list(updates.keys())}

@tool
def select_location(location_id: int, location_name: str) -> dict:
    """Selects a location from the options and updates the state."""
    return {"location": location_id, "location_name": location_name}

@tool
def select_provider(provider_id: int, provider_name: str) -> dict:
    """Selects a provider from the options and updates the state."""
    return {"provider": provider_id, "provider_name": provider_name}

@tool
def select_slot(start_time: str, operatory_id: int) -> dict:
    """Selects an appointment slot from the options and updates the state."""
    return {"slot_time": start_time, "operatory_id": operatory_id}

@tool
def user_confirms(confirmed: bool, message: Optional[str] = None) -> dict:
    """Registers the user's final decision (confirm or cancel) before booking."""
    return {"confirmed": confirmed, "message": message}




def inject_system_prompt(state: AppointmentState):
    """Injects the main system prompt and current date."""
    messages = state.messages
 
    if not messages or messages[0]["role"] != "system":
        today = datetime.now().strftime('%Y-%m-%d')
        system_prompt = (
            f"You are a helpful assistant for booking dental appointments. Today's date is {today}.\n"
            "1. Your first task is to collect the user's full name, phone number, and date of birth (YYYY-MM-DD). Use the `update_personal_info` tool to store this.\n"
            "2. Once you have name, phone, and DOB, **silently** (do not mention it) call `search_patient`.\n"
            "3. Based on the `search_patient` result, you'll know if they are an existing patient or a new patient. Continue the conversation accordingly.\n"
            "4. **Existing Patient Flow:** Welcome them back. Ask to book an appointment. Use `get_locations`, `get_providers`, and `get_slots` to find a time. Use `select_location`, `select_provider`, and `select_slot` to confirm their choices.\n"
            "5. **New Patient Flow:** Inform them they need to register. Ask for their email and use `update_personal_info`. Then, follow the same booking flow as an existing patient (get location, provider, slot). After they select a provider, call `create_patient` to register them *before* finding slots.\n"
            "6. **Slot Finding:** When the user asks for slots (e.g., 'tomorrow', 'next week', 'today after 3 pm'), use today's date ({today}) to calculate the correct `start_date` and `days` for the `get_slots` tool.\n"
            "7. **Confirmation:** Once a slot is selected and all info is present, clearly list *all* appointment details (Name, DOB, Phone, Email, Location, Provider, Time) and ask for final confirmation. Use the `user_confirms` tool.\n"
            "8. **Booking:** Only after `user_confirms(confirmed=True)` is called, you must call `make_appointment` to book it.\n"
            "Be conversational and helpful. Only ask for one piece of information at a time."
        )
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages

def call_llm_and_update_state(state: AppointmentState, tool_set: list) -> AppointmentState:
    """A generic function to call the LLM, execute tools, and update state."""
    
    
    messages = inject_system_prompt(state)
    
    llm_with_tools = llm.bind_tools(tool_set)
    response = llm_with_tools.invoke(messages)
    messages.append({"role": "assistant", "content": response.content})

    if tool_calls := getattr(response, "tool_calls", []):
        new_state = state.model_copy()
        for call in tool_calls:
            tool_name = call["name"]
            tool_args = call["args"]
           
            tool_map = {t.name: t for t in tool_set}
            if tool_name not in tool_map:
                messages.append({"role": "tool", "content": f"Error: Tool '{tool_name}' not found."})
                continue

            tool_func = tool_map[tool_name]
            try:
                result = tool_func.invoke(tool_args)
              
                if tool_name == "update_personal_info":
                    new_state = new_state.model_copy(update=result)
                elif tool_name == "search_patient":
                    if isinstance(result, list) and any(p.get("status") == "verified patient" for p in result):
                        p = result[0]
                        upcoming = p.get("upcoming_appts", [])
                        updates = {
                            "patient_id": str(p["id"]),
                            "is_existing_patient": True,
                            "location": p["location_ids"][0] if p.get("location_ids") else None,
                            "upcoming_appt_id": upcoming[0]["id"] if upcoming else None,
                            "provider": p.get("provider_id"),
                            "email": p.get("email")
                        }
                        new_state = new_state.model_copy(update=updates)
                    else:
                        new_state = new_state.model_copy(update={"is_existing_patient": False})
                elif tool_name in ["select_location", "select_provider", "select_slot"]:
                    new_state = new_state.model_copy(update=result)
                elif tool_name == "create_patient":
                    new_state = new_state.model_copy(update={"patient_id": str(result["id"])})
                elif tool_name in ["get_locations", "get_providers", "get_slots"]:
                    options_key = f"{tool_name.split('_')[1]}_options" 
                    new_state = new_state.model_copy(update={options_key: result})
                
                messages.append({"role": "tool", "tool_call_id": call['id'], "content": str(result)})

            except Exception as e:
                logging.error(f"Error calling tool {tool_name}: {e}")
                messages.append({"role": "tool", "tool_call_id": call['id'], "content": f"Error: {e}"})
        
        
        return new_state.model_copy(update={"messages": messages})

  
    return state.model_copy(update={"messages": messages})


def patient_info_node(state: AppointmentState) -> AppointmentState:
    """
    Collects basic patient info (name, phone, DOB) and determines if they
    are an existing or new patient by calling search_patient.
    """
    tools = [update_personal_info, search_patient, view_appointment]
    return call_llm_and_update_state(state, tools)

def new_patient_node(state: AppointmentState) -> AppointmentState:
    """
    Handles the flow for new patients:
    1. Collects email.
    2. Collects location and provider preference.
    3. Calls create_patient.
    4. Finds available slots.
    """
    tools = [
        update_personal_info,
        get_locations,
        get_providers,
        get_slots,
        select_location,
        select_provider,
        select_slot,
        create_patient
    ]
    return call_llm_and_update_state(state, tools)

def existing_patient_node(state: AppointmentState) -> AppointmentState:
    """
    Handles the flow for existing patients:
    1. Confirms/gets location and provider preference.
    2. Finds available slots.
    """
    tools = [
        get_locations,
        get_providers,
        get_slots,
        select_location,
        select_provider,
        select_slot,
        update_personal_info  
    ]
    return call_llm_and_update_state(state, tools)

def schedule_appointment_node(state: AppointmentState) -> AppointmentState:
    """
    Confirms all details with the user and books the appointment.
    """
    tools = [
        user_confirms,
        make_appointment,
        update_personal_info, 
        select_location,      
        select_provider,
    ]

    last_message = state.messages[-1]
    if last_message["role"] == "tool" and "confirmed" in last_message["content"]:
        
        import ast
        try:
            confirm_result = ast.literal_eval(last_message["content"])
            if confirm_result.get("confirmed"):
           
                try:
                    appt = make_appointment.invoke({
                        "patient_id": state.patient_id,
                        "location_id": state.location,
                        "provider_id": state.provider,
                        "operatory_id": state.operatory_id,
                        "start_time": state.slot_time,
                        "appointment_type_id": state.appointment_type_id
                    })
        
                    msg = (
                        f"Your appointment is booked!\n"
                        f"Appointment ID: {appt.get('appointment_id')}\n"
                        f"Time: {appt.get('start_time')}\n"
                        "Thank you for booking. Goodbye!"
                    )
                    messages = state.messages + [{"role": "assistant", "content": msg}]
                    return state.model_copy(update={"messages": messages})
                except Exception as e:
                    logging.error(f"Appointment booking failed: {e}")
                    msg = f"Error booking appointment: {e}. Please try again or update your details."
                    messages = state.messages + [{"role": "assistant", "content": msg}]
                    return state.model_copy(update={"messages": messages})
        except Exception as e:
            logging.error(f"Error parsing confirm_result: {e}")

    return call_llm_and_update_state(state, tools)



workflow = StateGraph(AppointmentState)


workflow.add_node("patient_info", patient_info_node)
workflow.add_node("new_patient", new_patient_node)
workflow.add_node("existing_patient", existing_patient_node)
workflow.add_node("schedule_appointment", schedule_appointment_node)


workflow.set_entry_point("patient_info")



def decide_patient_path(s: AppointmentState) -> str:
    """Routes from patient_info to the correct flow based on search results."""
    if s.is_existing_patient is True:
        return "existing_patient"
    if s.is_existing_patient is False:
        return "new_patient"
   
    return "patient_info"

def route_to_schedule(s: AppointmentState) -> str:
    """
    Checks if all required info for booking is present.
    For new patients, we also need a patient_id.
    """
    if s.slot_time and s.operatory_id and s.provider and s.location:
        if s.is_existing_patient:
            return "schedule_appointment"
        if not s.is_existing_patient and s.patient_id:
          
            return "schedule_appointment"
    

    return "new_patient" if not s.is_existing_patient else "existing_patient"

def decide_after_schedule(s: AppointmentState) -> str:
    """Decides what to do after the scheduling attempt."""
    last_msg = s.messages[-1]["content"]
    if "Your appointment is booked!" in last_msg:
        return END
    if "Booking canceled" in last_msg: 
        return END
    
    return "schedule_appointment"


workflow.add_conditional_edges("patient_info", decide_patient_path)
workflow.add_conditional_edges("new_patient", route_to_schedule)
workflow.add_conditional_edges("existing_patient", route_to_schedule)
workflow.add_conditional_edges("schedule_appointment", decide_after_schedule)


app = workflow.compile()

def run_interactive_workflow():
    state = AppointmentState(messages=[])
    print("Welcome to the Appointment Booking System! Type 'exit' to quit.")

    while True:
        user = input("You: ")
        if user.lower() == "exit":
            print("Goodbye!")
            break

      
        state.messages.append({"role": "user", "content": user})

        try:
          
            state = app.invoke(state)
            
         
            assistant_msg = state.messages[-1]
            if assistant_msg["role"] == "assistant" and assistant_msg["content"]:
                print(f"Assistant: {assistant_msg['content']}")
            
 
            if "Goodbye!" in assistant_msg["content"]:
                print("Session ended.")
                state = AppointmentState(messages=[]) 
                print("\n--- New Session ---")
                
        except Exception as e:
            print(f"Error: {e}")
            logging.error(f"Workflow error: {e}")
            state = AppointmentState(messages=[]) 
            print("An error occurred. Please start over.")

if __name__ == "__main__":
    run_interactive_workflow()