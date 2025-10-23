from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from datetime import datetime
from nexhealth_client import NexHealthClient
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)


class AppointmentState(BaseModel):
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    name: Optional[str] = None
    phone_number: Optional[str] = None
    dob: Optional[str] = None
    patient_id: Optional[str] = None
    location: Optional[int] = None
    provider: Optional[int] = None
    operatory_id: Optional[int] = None
    slot_time: Optional[str] = None
    appointment_type_id: Optional[int] = None
    is_existing_patient: Optional[bool] = None
    upcoming_appt_id: Optional[int] = None


@tool
def info_getter(name: str, phone_number: str, dob: str) -> dict:
    """Extract name, phone_number, and dob from user input."""
    return {"name": name, "phone_number": phone_number, "dob": dob}


@tool
def search_patient(name: str, phone_number: str, dob: str, location_id: int = 331668) -> Union[str, List[dict]]:
    """Search for a patient by name, phone, dob, and location."""
    client = NexHealthClient()
    return client.search_patients(name, phone_number, dob, location_id)


@tool
def get_locations() -> List[dict]:
    """Fetch all available locations from NexHealth."""
    return NexHealthClient().get_locations()


@tool
def get_providers(location_id: int) -> List[dict]:
    """Get list of providers for a given location_id."""
    return NexHealthClient().get_providers(location_id)


@tool
def get_slots(start_date: str, days: int, location_ids: List[int], provider_ids: List[int]) -> List[dict]:
    """Fetch available appointment slots."""
    client = NexHealthClient()
    return client.get_available_slots(start_date, days, location_ids, provider_ids)


@tool
def verify_appointment_data(
    name: Optional[str] = None,
    phone_number: Optional[str] = None,
    dob: Optional[str] = None,
    location: Optional[int] = None,
    provider: Optional[int] = None,
    operatory_id: Optional[int] = None,
    slot_time: Optional[str] = None,
) -> dict:
    """Check if all required fields are present."""
    input_data = {
        "name": name, "phone_number": phone_number, "dob": dob,
        "location": location, "provider": provider,
        "operatory_id": operatory_id, "slot_time": slot_time
    }
    required = ["name", "phone_number", "dob", "location", "provider", "operatory_id", "slot_time"]
    missing = [f for f in required if input_data.get(f) is None]
    return {"is_valid": not missing, "missing_fields": missing}


@tool
def confirmation_node(
    name: str,
    phone_number: str,
    dob: str,
    location: int,
    provider: int,
    slot_time: str,
    **kwargs
) -> str:
    """Generate confirmation message."""
    return (
        f"Please confirm:\n"
        f"Name: {name}\n"
        f"Phone: {phone_number}\n"
        f"DOB: {dob}\n"
        f"Location ID: {location}\n"
        f"Provider ID: {provider}\n"
        f"Slot: {slot_time}\n\n"
        f"Reply 'confirm' or 'cancel'"
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
    """Create a new appointment."""
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
def view_appointment(
    appointment_id: Optional[int] = None,
    location_id: Optional[int] = None,
    days: int = 10
) -> Union[str, dict, List[dict]]:
    """Retrieve appointment by ID or list upcoming ones."""
    client = NexHealthClient()
    return client.view_appointment(appointment_id, location_id, days)


@tool
def select_slot(slot_time: str, operatory_id: int) -> dict:
    """Call this tool when the user has definitively chosen a specific slot_time and operatory_id."""
    return {"slot_time": slot_time, "operatory_id": operatory_id}


@tool
def user_confirms(confirmed: bool, message: Optional[str] = None) -> dict:
    """
    Call this tool to register the user's final decision.
    Set 'confirmed' to True if they agreed.
    Set 'confirmed' to False if they canceled.
    'message' is for any final words, like 'Booking canceled.'
    """
    return {"confirmed": confirmed, "message": message}


def patient_info_node(state: AppointmentState) -> AppointmentState:
    llm_with_tools = llm.bind_tools([info_getter, search_patient, view_appointment])
    messages = state.messages

    if not messages or messages[0]["role"] != "system":
        system_prompt = (
            "You are a helpful assistant for booking appointments. "
            "Your first task is to get the user's full name, phone number, and date of birth (YYYY-MM-DD). "
            "Be conversational. If they only provide some information, acknowledge it and ask for what's missing. "
            "Once you have all three pieces of information, you *must* call the `info_getter` tool."
        )
        messages.insert(0, {"role": "system", "content": system_prompt})

    response = llm_with_tools.invoke(messages)
    tool_calls = getattr(response, "tool_calls", [])

    if tool_calls and tool_calls[0]["name"] == "info_getter":
        args = tool_calls[0]["args"]
        state = state.model_copy(update=args)

        search_result = search_patient.invoke({
            "name": state.name,
            "phone_number": state.phone_number,
            "dob": state.dob,
            "location_id": 331668
        })

        if isinstance(search_result, list) and any(p.get("status") == "verified patient" for p in search_result):
            p = search_result[0]
            upcoming = p.get("upcoming_appts", [])
            upcoming_id = upcoming[0]["id"] if upcoming else None
            prov_id = p.get("provider_id")

            print(p["location_ids"][0])

            state = state.model_copy(update={
                "patient_id": str(p["id"]),
                "is_existing_patient": True,
                "location": p["location_ids"][0],
                "upcoming_appt_id": upcoming_id,
                "provider": prov_id
            })
            msg = "Welcome back!"
            if upcoming_id:
                msg += f" You have an upcoming appointment (ID: {upcoming_id})."
            msg += " Let's book your next one."
        else:
            state = state.model_copy(update={"is_existing_patient": False})
            msg = "No record found. Let's register you as a new patient."

        messages.append({"role": "assistant", "content": msg})
        return state.model_copy(update={"messages": messages})

    if response.content:
        messages.append({"role": "assistant", "content": response.content})
        return state.model_copy(update={"messages": messages})

    messages.append({"role": "assistant", "content": "I'm sorry, I didn't catch that. Please provide your full name, phone, and date of birth."})
    return state.model_copy(update={"messages": messages})


def existing_patient_node(state: AppointmentState) -> AppointmentState:
    llm_with_tools = llm.bind_tools([get_providers, get_slots, select_slot])
    messages = state.messages
    
    system_prompt = (
        "Your task is now to book an appointment for this existing patient. "
        f"Their preferred location is {state.location}. "
    )
    
    if state.provider and not any("Here are the available providers" in m["content"] for m in messages):
        system_prompt += (
            f"Their preferred provider is {state.provider}. First, ask them if they want to use this provider. "
            "If they say yes, call `get_slots`. "
            "If they say no or want to see other options, call `get_providers`."
        )
    else:
        system_prompt += "The user wants to choose a provider. If you don't have a list, call `get_providers`. "
        system_prompt += "If they choose a provider ID, call `get_slots`. "
    
    system_prompt += "\nOnce they select a specific slot_time and operatory_id from the list, call `select_slot`."
    
    messages.append({"role": "system", "content": system_prompt})

    response = llm_with_tools.invoke(messages)
    messages.pop() 
    tool_calls = getattr(response, "tool_calls", [])

    if tool_calls:
        tool_call = tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        if tool_name == "get_providers":
            providers = get_providers.invoke(tool_args)
            prov_list = "\n".join([f"{p['id']}: {p['name']}" for p in providers])
            msg = f"Here are the available providers at your location:\n{prov_list}\nPlease select an ID."
            messages.append({"role": "assistant", "content": msg})
            return state.model_copy(update={"messages": messages, "provider": None})

        elif tool_name == "get_slots":
            if "provider_ids" not in tool_args or not tool_args["provider_ids"]:
                tool_args["provider_ids"] = [state.provider]
            tool_args["location_ids"] = [state.location]
            if "start_date" not in tool_args:
                tool_args["start_date"] = datetime.now().strftime("%Y-%m-%d")
            if "days" not in tool_args:
                tool_args["days"] = 7
                
            slots = get_slots.invoke(tool_args)
            if not slots:
                messages.append({"role": "assistant", "content": "I'm sorry, no slots are available for that provider in the next 7 days. Would you like to change providers?"})
                return state.model_copy(update={"messages": messages, "provider": None}) 
                
            slot_list = "\n".join([f"Time: {s['start_time']} (Operatory ID: {s['operatory_id']})" for s in slots])
            msg = f"Here are the available slots:\n{slot_list}\nPlease select a time and its operatory ID."
            messages.append({"role": "assistant", "content": msg})
            state = state.model_copy(update={"provider": tool_args["provider_ids"][0]})

        elif tool_name == "select_slot":
            selected = tool_call["args"]
            messages.append({"role": "assistant", "content": f"Great, I've selected {selected['slot_time']}."})
            return state.model_copy(update={
                "slot_time": selected["slot_time"],
                "operatory_id": selected["operatory_id"],
                "messages": messages
            })

    else:
        messages.append({"role": "assistant", "content": response.content})

    return state.model_copy(update={"messages": messages})


def new_patient_node(state: AppointmentState) -> AppointmentState:
    llm_with_tools = llm.bind_tools([get_locations, get_providers, get_slots, select_slot])
    messages = state.messages

    system_prompt = (
        "Your task is to register a new patient. "
        "First, you *must* call `get_locations` to show them the options. "
        "Once they choose a location ID, you *must* call `get_providers` for that location. "
        "Once they choose a provider ID, you *must* call `get_slots`. "
        "Once they select a specific slot_time and operatory_id, call `select_slot`."
    )
    messages.append({"role": "system", "content": system_prompt})
        
    response = llm_with_tools.invoke(messages)
    messages.pop()
    tool_calls = getattr(response, "tool_calls", [])

    if tool_calls:
        tool_call = tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "get_locations":
            locs = get_locations.invoke(tool_args)
            loc_list = "\n".join([f"{l['id']}: {l['name']}, {l['city']}" for l in locs])
            msg = f"Here are our locations:\n{loc_list}\nPlease select an ID."
            messages.append({"role": "assistant", "content": msg})
            return state.model_copy(update={"messages": messages})

        elif tool_name == "get_providers":
            if "location_id" not in tool_args:
                messages.append({"role": "assistant", "content": "I'm sorry, I didn't catch which location you wanted. Please select a location ID from the list."})
                return state.model_copy(update={"messages": messages})
                
            provs = get_providers.invoke(tool_args)
            prov_list = "\n".join([f"{p['id']}: {p['name']}" for p in provs])
            msg = f"Great. Here are the providers at that location:\n{prov_list}\nPlease select an ID."
            messages.append({"role": "assistant", "content": msg})
            return state.model_copy(update={"messages": messages, "location": tool_args["location_id"]})

        elif tool_name == "get_slots":
            if "provider_ids" not in tool_args or not state.location:
                messages.append({"role": "assistant", "content": "I'm sorry, I'm missing a provider. Please select a provider ID."})
                return state.model_copy(update={"messages": messages})
            
            tool_args["location_ids"] = [state.location]
            if "start_date" not in tool_args:
                tool_args["start_date"] = datetime.now().strftime("%Y-%m-%d")
            if "days" not in tool_args:
                tool_args["days"] = 7

            slots = get_slots.invoke(tool_args)
            if not slots:
                messages.append({"role": "assistant", "content": "I'm sorry, no slots are available for that provider in the next 7 days. Would you like to change providers?"})
                return state.model_copy(update={"messages": messages, "provider": None})

            slot_list = "\n".join([f"Time: {s['start_time']} (Operatory ID: {s['operatory_id']})" for s in slots])
            msg = f"Here are the available slots:\n{slot_list}\nPlease select a time and its operatory ID."
            messages.append({"role": "assistant", "content": msg})
            return state.model_copy(update={"messages": messages, "provider": tool_args["provider_ids"][0]})

        elif tool_name == "select_slot":
            selected = tool_call["args"]
            messages.append({"role": "assistant", "content": f"Great, I've selected {selected['slot_time']}."})
            return state.model_copy(update={
                "slot_time": selected["slot_time"],
                "operatory_id": selected["operatory_id"],
                "messages": messages
            })
            
    else:
        messages.append({"role": "assistant", "content": response.content})

    return state.model_copy(update={"messages": messages})


def schedule_appointment_node(state: AppointmentState) -> AppointmentState:
    llm_with_tools = llm.bind_tools([verify_appointment_data, confirmation_node, user_confirms])
    messages = state.messages

    if not any("Please confirm" in m["content"] for m in messages):
        ver_input = {
            "name": state.name, "phone_number": state.phone_number, "dob": state.dob,
            "location": state.location, "provider": state.provider,
            "operatory_id": state.operatory_id, "slot_time": state.slot_time
        }
        ver = verify_appointment_data.invoke(ver_input)

        if not ver["is_valid"]:
            msg = f"I'm almost ready, but I'm still missing: {', '.join(ver['missing_fields'])}. Can you provide that?"
            messages.append({"role": "assistant", "content": msg})
            return state.model_copy(update={"messages": messages})

        conf_msg = confirmation_node.invoke(ver_input)
        messages.append({"role": "assistant", "content": conf_msg})
        return state.model_copy(update={"messages": messages})

    messages.append({"role": "system", "content": "The user is now replying to the confirmation message. Use the `user_confirms` tool to capture their 'yes' or 'no' response."})
    response = llm_with_tools.invoke(messages)
    messages.pop()
    tool_calls = getattr(response, "tool_calls", [])

    if tool_calls and tool_calls[0]["name"] == "user_confirms":
        decision = tool_calls[0]["args"]
        
        if decision["confirmed"]:
            try:
                appt = make_appointment.invoke({
                    "patient_id": state.patient_id or "new_patient",
                    "location_id": state.location,
                    "provider_id": state.provider,
                    "operatory_id": state.operatory_id,
                    "start_time": state.slot_time,
                    "appointment_type_id": 1
                })
                msg = f"All set! Your appointment is booked. The ID is: {appt.get('appointment_id', 'N/A')}"
                messages.append({"role": "assistant", "content": msg})
            except Exception as e:
                logging.error(f"Appointment booking failed: {e}")
                messages.append({"role": "assistant", "content": f"I'm sorry, something went wrong while booking. Error: {e}"})
            
            return state.model_copy(update={"messages": messages})
        
        else:
            msg = decision.get("message", "Booking canceled.")
            messages.append({"role": "assistant", "content": msg})
            return state.model_copy(update={"messages": messages})

    if not tool_calls and response.content:
        messages.append({"role": "assistant", "content": response.content})
    
    return state.model_copy(update={"messages": messages})


workflow = StateGraph(AppointmentState)
workflow.add_node("patient_info", patient_info_node)
workflow.add_node("existing_patient", existing_patient_node)
workflow.add_node("new_patient", new_patient_node)
workflow.add_node("schedule_appointment", schedule_appointment_node)

workflow.set_entry_point("patient_info")

def decide_patient_path(s: AppointmentState):
    """Decide where to go after patient_info node."""
    if s.is_existing_patient is True:
        return "existing_patient"
    if s.is_existing_patient is False:
        return "new_patient"
    return END

workflow.add_conditional_edges("patient_info", decide_patient_path)

def route_to_schedule(s: AppointmentState):
    """If slot is picked, go to schedule. Otherwise, stay in patient node."""
    if s.slot_time:
        return "schedule_appointment"
    
    if s.is_existing_patient:
        return "existing_patient"
    return "new_patient"

workflow.add_conditional_edges("existing_patient", route_to_schedule)
workflow.add_conditional_edges("new_patient", route_to_schedule)


def decide_schedule_path(s: AppointmentState):
    """Decide if booking is done or needs more info."""
    last_msg = s.messages[-1]["content"].lower()
    if "booked" in last_msg or "canceled" in last_msg:
        return END
    
    if "missing" in last_msg:
        return "patient_info"
        
    return "schedule_appointment"

workflow.add_conditional_edges("schedule_appointment", decide_schedule_path)

app = workflow.compile()


def run_interactive_workflow():
    state = {} 
    print("Welcome to the Appointment Booking System! Type 'exit' to quit.")
    
    while True:
        user = input("You: ")
        if user.lower() == "exit":
            print("Goodbye!")
            break

        current_messages = state.get("messages", []) + [{"role": "user", "content": user}]
        state = {**state, "messages": current_messages}

        try:
            state = app.invoke(state) 
            
            assistant = state.get("messages", [])[-1]["content"]
            print(f"Assistant: {assistant}")
            
            if "booked" in assistant.lower() or "canceled" in assistant.lower():
                print("Session ended. Start a new one.")
                state = {}
        except Exception as e:
            print(f"Error: {e}")
            state = {}
            print("Please start over with name, phone, and DOB.")


if __name__ == "__main__":
    run_interactive_workflow()