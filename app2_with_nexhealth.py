"""
clinic_agent_langgraph_validated.py
-----------------------------------
Enhanced version with NexHealth integration while maintaining original flow
"""

import json
import re
import os
import time
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from nexhealth_client import NexHealthClient
from datetime import datetime, timedelta
from dotenv import load_dotenv


load_dotenv()

# --------------------------------------------------
# KNOWLEDGE BASE (Unchanged)
# --------------------------------------------------
KB = {
    "hours": "We are open Monday through Friday from 9 a.m. to 5 p.m.",
    "location": "We are at 123 Health Street, Suite 200.",
    "insurance": "We accept most major insurance plans, including Aetna, Blue Cross, and United.",
    "phone": "You can call the front desk at 555-123-4567."
}

def kb_answer(message: str) -> str:
    text = message.lower()
    if "hour" in text or "open" in text or "time" in text:
        return KB["hours"]
    if "where" in text or "location" in text or "address" in text:
        return KB["location"]
    if "insurance" in text or "accept" in text:
        return KB["insurance"]
    if "phone" in text or "number" in text or "call" in text:
        return KB["phone"]
    return "I'm not sure about that. I can help with hours, insurance, our location, or appointments."

# --------------------------------------------------
# NEXHEALTH DATA FUNCTIONS
# --------------------------------------------------
def normalize_phone(p: str) -> str:
    digits = re.sub(r"\D", "", p)
    return digits if len(digits) == 10 else ""

def valid_dob(d: str) -> bool:
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", d))

def find_patient(name: str, dob: str, phone: str) -> bool:
    client = NexHealthClient()
    result = client.search_patients(name, phone, dob, location_id=331668)
    if isinstance(result, list) and len(result) > 0:
        return True
    return False

def create_patient_record(name: str, dob: str, phone: str, email: str):
    client = NexHealthClient()
 
    parts = name.split(None, 1)
    first_name = parts[0]
    last_name = parts[1] if len(parts) > 1 else ""
    
    result = client.create_patient(
        provider_id=413326781,  
        first_name=first_name,
        last_name=last_name,
        email=email,
        phone_number=phone,
        date_of_birth=dob,
        location_id=331668  
    )
    return result

def check_availability(provider: str, date: str) -> bool:
    client = NexHealthClient()

    provider_id = get_provider_id_by_name(provider)
    if not provider_id:
        return False
        
    slots = client.get_available_slots(
        start_date=date,
        days=1,
        location_ids=[331668],
        provider_ids=[provider_id]
    )
    return len(slots) > 0

def get_provider_id_by_name(provider_name: str) -> Optional[int]:
    client = NexHealthClient()
    providers = client.get_providers(location_id=331668)
    for p in providers:
        if provider_name.lower() in p['name'].lower():
            return p['id']
    return None

def schedule_appointment(provider: str, date: str, patient_name: str) -> str:
    client = NexHealthClient()
    provider_id = get_provider_id_by_name(provider)
    if not provider_id:
        return f"Could not find provider {provider}"
        
  
    patient_info = client.search_patients(patient_name, "", "", 331668)
    if not isinstance(patient_info, list) or not patient_info:
        return "Patient not found"
        
    patient_id = str(patient_info[0]['id'])
  
    slots = client.get_available_slots(
        start_date=date,
        days=1,
        location_ids=[331668],
        provider_ids=[provider_id]
    )
    
    if not slots:
        return f"No slots available for {provider} on {date}"

    slot = slots[0]
    result = client.make_appointment(
        patient_id=patient_id,
        location_id=331668,
        provider_id=provider_id,
        operatory_id=slot['operatory_id'],
        start_time=slot['start_time'],
        appointment_type_id=1
    )
    
    if result.get('error'):
        return f"Booking failed: {result['error']}"
        
    return f"All set. {patient_name} is booked with {provider} on {date}"

# --------------------------------------------------
# STATE
# --------------------------------------------------
class State(BaseModel):
    user_message: str
    intent: Optional[str] = None
    patient_name: Optional[str] = None
    dob: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    provider: Optional[str] = None
    date: Optional[str] = None
    confirmed: bool = False
    result: Optional[str] = None

# --------------------------------------------------
# LangChain + LangGraph
# --------------------------------------------------
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="google/gemini-2.5-flash-lite",
    default_headers={
        # "HTTP-Referer": os.getenv("YOUR_SITE_URL"),
        # "X-Title": os.getenv("YOUR_SITE_NAME"),
    }
)

def intent_node(state: State) -> State:
    messages = [
        SystemMessage(content="Keep answers short, friendly, and easy to speak."),
        HumanMessage(content=f"""
User: "{state.user_message}"
Classify intent as ONLY:
- "general_question"
- "appointment_request"
Return JSON: {{"intent":"<value>"}}
""")
    ]
    res = llm.invoke(messages).content
    try:
        data = json.loads(res)
        state.intent = data.get("intent")
    except:
        state.intent = "general_question"
    return state

def route_from_intent(state: State) -> str:
    return "ANSWER_KB" if state.intent == "general_question" else "PARSE"

def answer_kb_node(state: State) -> State:
    state.result = kb_answer(state.user_message)
    return state

def parse_node(state: State) -> State:
    messages = [
        SystemMessage(content="Extract information from user messages."),
        HumanMessage(content=f"""
Extract fields from this message:
"{state.user_message}"
Return JSON:
{{
 "patient_name": <string or null>,
 "dob": <string or null>,
 "phone": <string or null>,
 "email": <string or null>,
 "provider": <string or null>,
 "date": <string or null>
}}
""")
    ]
    raw = llm.invoke(messages).content
    try:
        data = json.loads(raw)
    except:
        data = {}

    state.patient_name = state.patient_name or data.get("patient_name")
    state.dob = state.dob or data.get("dob")
    state.phone = state.phone or data.get("phone")
    state.email = state.email or data.get("email")
    state.provider = state.provider or data.get("provider")
    state.date = state.date or data.get("date")
    return state

def collect_info_node(state: State) -> State:
  
    if not state.patient_name:
        state.result = "Sure. What's the full name of the patient?"
        return state

    
    if not state.dob:
        state.result = "What is the date of birth? Please use YYYY dash MM dash DD."
        return state
    if not valid_dob(state.dob):
        state.dob = None
        state.result = "The date of birth looks off. Please use YYYY dash MM dash DD."
        return state

    if not state.phone:
        state.result = "What is the best phone number to reach the patient?"
        return state
    if not normalize_phone(state.phone):
        state.phone = None
        state.result = "That phone number seems incomplete. Please give 10 digits."
        return state

    if not state.email:
        state.result = "What's your email address?"
        return state

    return state

def needs_more_info(state: State) -> bool:
    if not state.patient_name or not state.dob or not state.phone or not state.email:
        return True
    return False

def check_patient_node(state: State) -> State:
    if not find_patient(state.patient_name, state.dob, state.phone):
        create_patient_record(state.patient_name, state.dob, state.phone, state.email)
        state.result = f"Thanks {state.patient_name}. You're set in our system. Who would you like to see, and on what date?"
        return state
    return state

def availability_node(state: State) -> State:
    if not state.provider or not state.date:
        state.result = "Which provider and what date works for you?"
        return state

    if not check_availability(state.provider, state.date):
        client = NexHealthClient()
        provider_id = get_provider_id_by_name(state.provider)
        if provider_id:
            slots = client.get_available_slots(
                start_date=datetime.now().strftime("%Y-%m-%d"),
                days=7,
                location_ids=[331668],
                provider_ids=[provider_id]
            )
            if slots:
                alt_date = datetime.fromisoformat(slots[0]['start_time']).strftime("%Y-%m-%d")
                state.result = f"{state.provider} is booked on that date. {alt_date} is open. Should I use that instead?"
            else:
                state.result = f"{state.provider} has no openings. Want a different provider?"
        return state

    return state

def schedule_node(state: State) -> State:
    if not state.confirmed:
        state.result = f"Just to confirm, book {state.patient_name} with {state.provider} on {state.date}?"
        return state

    state.result = schedule_appointment(state.provider, state.date, state.patient_name)
    return state

# --------------------------------------------------
# BUILD GRAPH 
# --------------------------------------------------
def build_graph():
    graph = StateGraph(State)

    graph.add_node("INTENT", intent_node)
    graph.add_node("ANSWER_KB", answer_kb_node)
    graph.add_node("PARSE", parse_node)
    graph.add_node("COLLECT_INFO", collect_info_node)
    graph.add_node("CHECK_PATIENT", check_patient_node)
    graph.add_node("CHECK_AVAIL", availability_node)
    graph.add_node("SCHEDULE", schedule_node)

    # graph.set_entry_point("INTENT")

    # graph.add_conditional_edges(
    #     "INTENT",
    #     route_from_intent,
    #     {
    #         "ANSWER_KB": "ANSWER_KB",
    #         "PARSE": "PARSE"
    #     }
    # )

    graph.set_entry_point("COLLECT_INFO")

    # graph.add_edge("PARSE", "COLLECT_INFO")

    graph.add_conditional_edges(
        "COLLECT_INFO",
        lambda s: "COLLECT_INFO" if needs_more_info(s) else "CHECK_PATIENT",
        {
            "COLLECT_INFO": "COLLECT_INFO",
            "CHECK_PATIENT": "CHECK_PATIENT"
        }
    )

    graph.add_edge("CHECK_PATIENT", "CHECK_AVAIL")
    graph.add_edge("CHECK_AVAIL", "SCHEDULE")
    graph.add_edge("SCHEDULE", END)

    return graph.compile()

# --------------------------------------------------
# CHAT LOOP
# --------------------------------------------------
if __name__ == "__main__":
    app = build_graph()
    print("Clinic Assistant ready. Just talk normally.\n")

    current_state = None

    while True:
        user = input("\nYou: ")
        if user.lower() in ["quit", "exit", "stop"]:
            print("Assistant: Goodbye. Take care.")
            break

        # Create or update state
        if current_state is None:
            current_state = State(user_message=user)
        else:
            current_state = State(
                user_message=user,
                intent=current_state.intent,
                patient_name=current_state.patient_name,
                dob=current_state.dob,
                phone=current_state.phone,
                email=current_state.email,
                provider=current_state.provider,
                date=current_state.date,
                confirmed=True if user.lower() in ["yes", "yeah", "yep", "confirm", "sure"] else current_state.confirmed
            )

        result = app.invoke(current_state)
       
        current_state = State(**result)
        print("Assistant:", current_state.result)

       
        if current_state.result and "booked" in current_state.result.lower():
            current_state = None