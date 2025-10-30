import os
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from datetime import datetime, timedelta
from nexhealth_client import NexHealthClient
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import logging
import time  

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="google/gemini-2.5-flash-lite",
    default_headers={
        # "HTTP-Referer": os.getenv("YOUR_SITE_URL"),
        # "X-Title": os.getenv("YOUR_SITE_NAME"),
    }
)

class AppointmentState(BaseModel):
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    name: Optional[str] = None
    phone_number: Optional[str] = None
    dob: Optional[str] = None
    email: Optional[str] = None 
    patient_id: Optional[str] = None

    location: Optional[int] = None
    location_name: Optional[str] = None
    provider: Optional[int] = None
    provider_name: Optional[str] = None
    
    operatory_id: Optional[int] = None
    slot_time: Optional[str] = None
    appointment_type_id: Optional[int] = 1 
    
    is_existing_patient: Optional[bool] = None
    upcoming_appt_id: Optional[int] = None

    location_options: Optional[List[Dict]] = Field(default_factory=list)
    provider_options: Optional[List[Dict]] = Field(default_factory=list)
    slot_options: Optional[List[Dict]] = Field(default_factory=list)

    # New fields for improvements
    preferred_provider: Optional[int] = None
    preferred_provider_name: Optional[str] = None
    awaiting_confirm: Optional[bool] = None


@tool
def info_getter(name: str, phone_number: str, dob: str) -> dict:
    """Extract name, phone_number, and dob from the conversation."""
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
    """Check if all required fields (IDs) are present."""
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
    location_name: Optional[str] = None,
    provider_name: Optional[str] = None,
    **kwargs
) -> str:
    """Generate confirmation message using names."""

    loc_display = location_name if location_name else f"ID: {location}"
    prov_display = provider_name if provider_name else f"ID: {provider}"

    try:
        time_obj = datetime.fromisoformat(slot_time)
        slot_display = time_obj.strftime('%A, %B %d, %Y at %I:%M %p')
    except Exception:
        slot_display = slot_time

    return (
        f"Please confirm your appointment:\n"
        f"Name: {name}\n"
        f"Phone: {phone_number}\n"
        f"DOB: {dob}\n"
        f"Location: {loc_display}\n"
        f"Provider: {prov_display}\n"
        f"Slot: {slot_display}\n\n"
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
    appointment_created=client.make_appointment(
        patient_id,
        location_id,
        provider_id,
        operatory_id,
        start_time,
        appointment_type_id
    )
    error=appointment_created.get("error")
    if error is not None:
        return {"error": error}

    return appointment_created


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
def select_slot(selection: str) -> dict:
    """
    Call this tool when the user has definitively chosen a specific slot from the list.
    'selection' can be the number (e.g., '1', '2') or a string matching the time.
    """
    return {"selection": selection}


@tool
def user_confirms(confirmed: bool, message: Optional[str] = None) -> dict:
    """
    Call this tool to register the user's final decision.
    Set 'confirmed' to True if they agreed.
    Set 'confirmed' to False if they canceled.
    'message' is for any final words, like 'Booking canceled.'
    """
    return {"confirmed": confirmed, "message": message}




@tool
def get_email(email: str) -> dict:
    """Call this to capture the user's email address."""
    return {"email": email}


@tool
def select_provider(selection: str) -> dict:
    """
    Call this tool when the user has definitively chosen a specific provider from the list.
    'selection' can be the number (e.g., '1', '2') or a string matching the name.
    """
    return {"selection": selection}


@tool
def create_patient(
    provider_id: int,
    full_name: str,
    email: str,
    phone_number: str,
    date_of_birth: str,
    location_id: int
) -> dict:
    """
    Creates a new patient.
    full_name will be split into first_name and last_name.
    Returns the new patient's details, including their new ID.
    """

    parts = full_name.split(None, 1)
    first_name = parts[0]
    last_name = parts[1] if len(parts) > 1 else ""
    
    client = NexHealthClient()
    patient_data = client.create_patient(
        provider_id=provider_id,
        first_name=first_name,
        last_name=last_name,
        email=email,
        phone_number=phone_number,
        date_of_birth=date_of_birth,
        location_id=location_id
    )

    error=patient_data.get("error")
    if error is not None:
        return {"error": error}
    
    patient_data["patient_id"] = patient_data.get("id")
    return patient_data


@tool
def select_location(selection: str) -> dict:
    """
    Call this tool when the user has definitively chosen a specific location from the list.
    'selection' can be the number (e.g., '1', '2') or a string matching the name.
    """
    return {"selection": selection}



def _get_llm_response(messages: List[Dict], tools: List):
    """Helper function to invoke LLM with retry."""
    llm_with_tools = llm.bind_tools(tools)
    try:
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        if "429" in str(e):
            logging.warning("Quota hit; waiting 60s...")
            time.sleep(60)
            response = llm_with_tools.invoke(messages) 
        else:
            raise e
    return response


def patient_info_node(state: AppointmentState) -> AppointmentState:
    if state.is_existing_patient is not None:
        return state

    messages = state.messages
    if not messages or messages[0]["role"] != "system":
        system_prompt = (
            "You are a helpful assistant for booking appointments at my Health Clinic. "
            "Your first task is to get the user's full name, phone number, and date of birth (YYYY-MM-DD). "
            "Be conversational. If they only provide some information, acknowledge it and ask for what's missing. "
            "You can accumulate information across messages. "
            "Once you have all three pieces of information from the conversation, you *must* call the `info_getter` tool with the extracted values."
        )
        messages.insert(0, {"role": "system", "content": system_prompt})

    response = _get_llm_response(messages, [info_getter, search_patient, view_appointment])
    content = getattr(response, "content", "")
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
            loc_id = p["location_ids"][0]
            prov_id = p.get("provider_id")
            
            loc_name = f"ID: {loc_id}"
            prov_name = f"ID: {prov_id}" if prov_id else "any provider"
            try:
                all_locs = get_locations.invoke({})
                found_loc = next((loc for loc in all_locs if loc['id'] == loc_id), None)
                if found_loc:
                    loc_name = found_loc['name']
                
                if prov_id:
                    all_provs = get_providers.invoke({"location_id": loc_id})
                    found_prov = next((prov for prov in all_provs if prov['id'] == prov_id), None)
                    if found_prov:
                        prov_name = found_prov['name']
            except Exception as e:
                logging.warning(f"Could not fetch names for existing patient: {e}")
    

            state = state.model_copy(update={
                "patient_id": str(p["id"]),
                "is_existing_patient": True,
                "location": loc_id,
                "location_name": loc_name,
                "upcoming_appt_id": upcoming_id,
                "preferred_provider": prov_id,
                "preferred_provider_name": prov_name,
                "awaiting_confirm": True
            })
            
            msg = f"Welcome back, {state.name}! Your preferred location is {loc_name}."
            if upcoming_id:
                msg += f" I see you have an upcoming appointment (ID: {upcoming_id})."
            
            msg += " Let's book your next appointment."
            if prov_id:
                msg += f" Your preferred provider is {prov_name}. Would you like to book at this location with this provider?"
            else:
                msg += " Would you like to book at this location? We'll need to select a provider."
        
        else:
            state = state.model_copy(update={"is_existing_patient": False})
            msg = "Thanks! I don't see a record for you. Let's register you as a new patient."

        messages.append({"role": "assistant", "content": msg})
        return state.model_copy(update={"messages": messages})

    if not content:
        content = "I'm sorry, I didn't catch that. Please provide your full name, phone, and date of birth."
        
    messages.append({"role": "assistant", "content": content})
    return state.model_copy(update={"messages": messages})


def _format_slots_for_user(slots: List[Dict]) -> str:
    """Helper to format slots into a user-friendly numbered list."""
    if not slots:
        return "No slots available."
    
    formatted = []
    for i, s in enumerate(slots):
        try:
            time_obj = datetime.fromisoformat(s['start_time'])
            time_str = time_obj.strftime('%A, %B %d at %I:%M %p') 
            formatted.append(f"{i+1}. {time_str}")
        except Exception:
            formatted.append(f"{i+1}. {s['start_time']}")
            
    return "\n".join(formatted)


def _find_selected_slot(selection: str, slot_options: List[Dict]) -> Optional[Dict]:
    """Helper to find a slot from options based on user's selection (number or text)."""
    if not slot_options:
        return None
        

    if selection.isdigit():
        idx = int(selection) - 1
        if 0 <= idx < len(slot_options):
            return slot_options[idx]
            
  
    selection_lower = selection.lower()
    for s in slot_options:
        if selection_lower in s['start_time'].lower():
            return s
            

    for i, s in enumerate(slot_options):
        try:
            time_obj = datetime.fromisoformat(s['start_time'])
            time_str_1 = time_obj.strftime('%A, %B %d at %I:%M %p').lower()
            time_str_2 = time_obj.strftime('%I:%M %p').lower()
            if selection_lower in time_str_1 or selection_lower in time_str_2:
                return s
        except Exception:
            continue
            
    logging.warning(f"Could not match selection '{selection}' to any slot.")
    return None


def _find_selected_provider(selection: str, provider_options: List[Dict]) -> Optional[Dict]:
    """Helper to find a provider from options based on user's selection (number or text)."""
    if not provider_options:
        return None
        
    if selection.isdigit():
        idx = int(selection) - 1
        if 0 <= idx < len(provider_options):
            return provider_options[idx]
            
    selection_lower = selection.lower()
    for p in provider_options:
        if selection_lower in p['name'].lower():
            return p
            
    logging.warning(f"Could not match selection '{selection}' to any provider.")
    return None


def _find_selected_location(selection: str, location_options: List[Dict]) -> Optional[Dict]:
    """Helper to find a location from options based on user's selection (number or text)."""
    if not location_options:
        return None
        
    if selection.isdigit():
        idx = int(selection) - 1
        if 0 <= idx < len(location_options):
            return location_options[idx]
            
    selection_lower = selection.lower()
    for l in location_options:
        if selection_lower in l['name'].lower():
            return l
            
    logging.warning(f"Could not match selection '{selection}' to any location.")
    return None


def existing_patient_node(state: AppointmentState) -> AppointmentState:
    if state.slot_time:  
        return state

    messages = state.messages
    system_prompts = []
    tools = []
    
    if state.awaiting_confirm:
        system_prompts.append(
            "The user is responding to the confirmation of location and provider. "
            "If they confirm everything, ask for a start date to search for slots. "
            f"Today's date is {datetime.now().strftime('%Y-%m-%d')}. "
            "Convert their answer (e.g.,'today', 'tomorrow', 'next week') into a 'YYYY-MM-DD' `start_date`. "
            "Adjust 'days' based on request (default 1, e.g., 2 for '2 days', 14 for 'two weeks' for today the days are like 1 etc). "
            "If time filters (e.g., 'after 5pm'), include 'min_time': '17:00' and/or 'max_time': 'HH:MM'. "
            "Then, call `get_slots` with `start_date`, `days`, `location_ids`=[{state.location}], `provider_ids`=[{state.preferred_provider}]. "
            "If they want a different provider, call `get_providers(location_id={state.location})`. "
            "If they want a different location, call `get_locations()`. "
        )
        tools = [get_slots, get_providers, get_locations]
    
    if state.provider_options and not state.provider:

        prov_list_str = "\n".join([f"Number {i+1} ('{p['name']}') is ID {p['id']}" for i, p in enumerate(state.provider_options)])
        system_prompts.append(
            "The user was shown a list of providers. They will now select one by name or number. "
            f"You must map their selection to the correct provider ID from this list:\n{prov_list_str}\n"
            f"Once you have the provider ID, ask them for a start date to search for slots. Today's date is {datetime.now().strftime('%Y-%m-%d')}. "
            "You MUST convert their answer (e.g., 'tomorrow', 'today') into a 'YYYY-MM-DD' `start_date`. "
            "Then, you *must* call `get_slots` with that `start_date`, `days=7`, and the `provider_ids` list."
        )
        tools = [get_slots]

    elif state.slot_options and not state.slot_time:

        slot_list_str = "\n".join([f"Number {i+1} is '{s['start_time']}' (Operatory: {s['operatory_id']})" for i, s in enumerate(state.slot_options)])
        system_prompts.append(
            "The user was shown a list of slots. They will now select one by number or time. "
            f"You must capture their selection as a string (e.g., '1' or '9am'). Here is the list for your reference:\n{slot_list_str}\n"
            "If the user asks for the first one, earliest, or just one slot, use '1'."
            "You *must* call `select_slot(selection=...)` with their choice."
        )
        tools = [select_slot]
        
    elif state.location_options and not state.location:
        loc_list_str = "\n".join([f"Number {i+1} ('{l['name']}') is ID {l['id']}" for i, l in enumerate(state.location_options)])
        system_prompts.append(
            "The user was shown a list of locations. They will now select one by name or number. "
            f"You must capture their selection as a string (e.g., '1' or 'Location Name'). Here is the list for your reference:\n{loc_list_str}\n"
            "You *must* call `select_location(selection=...)` with their choice."
        )
        tools = [select_location]

    elif state.provider_options and not state.provider:
        prov_list_str = "\n".join([f"Number {i+1} ('{p['name']}') is ID {p['id']}" for i, p in enumerate(state.provider_options)])
        system_prompts.append(
            "The user was shown a list of providers. They will now select one by name or number. "
            f"You must capture their selection as a string (e.g., '1' or 'Dr. Smith'). Here is the list for your reference:\n{prov_list_str}\n"
            "You *must* call `select_provider(selection=...)` with their choice."
        )
        tools = [select_provider]

    elif not state.provider:
     
        system_prompts.append(
            "The patient needs to select a provider. "
            f"Their preferred provider is {state.preferred_provider_name} (ID: {state.preferred_provider}). "
            "First, ask them if they want to use this provider. "
            "If they say yes, ask them for a start date to search for slots. "
            f"Today's date is {datetime.now().strftime('%Y-%m-%d')}. "
            "You MUST convert their answer (e.g.,'today', 'tomorrow') into a 'YYYY-MM-DD' `start_date`. "
            "Adjust 'days' based on request (default 1, e.g., 2 for '2 days', 14 for 'two weeks' for today the days are like 1 etc). "
            "If time filters (e.g., 'after 5pm'), include 'min_time': '17:00' and/or 'max_time': 'HH:MM'. "
            "Then, you *must* call `get_slots` with that `start_date`, `days`, and `provider_ids`=[{state.preferred_provider}]. "
            "If they say no or want other options, call `get_providers`."
        )
        tools = [get_providers, get_slots]
    
    else:
      
        system_prompts.append(
            "The user has confirmed their provider. Ask them for a start date to search for slots. "
            f"Today's date is {datetime.now().strftime('%Y-%m-%d')}. "
            "You MUST convert their answer (e.g.,'today',  'tomorrow', 'next week') into a 'YYYY-MM-DD' `start_date`. "
            "Adjust 'days' based on request (default 1, e.g., 2 for '2 days', 14 for 'two weeks' for today the days are like 1 etc). "
            "If time filters (e.g., 'after 5pm'), include 'min_time': '17:00' and/or 'max_time': 'HH:MM'. "
            "Do NOT ask for the number of days. "
            f"Then, you *must* call `get_slots` with that `start_date`, `days`, `location_ids`=[{state.location}], and `provider_ids`=[{state.provider}]."
        )
        tools = [get_slots]

   
    if messages[-1]["role"] != "system":
        messages.append({"role": "system", "content": "\n".join(system_prompts)})
    
    response = _get_llm_response(messages, tools)
    content = getattr(response, "content", "")
    tool_calls = getattr(response, "tool_calls", [])


    if messages[-1]["role"] == "system":
        messages.pop()

    if tool_calls:
        tool_call = tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        if tool_name == "get_locations":
            locations = get_locations.invoke(tool_args)
            loc_list = "\n".join([f"{i+1}. {l['name']}" for i, l in enumerate(locations)])
            msg = f"Here are the available locations:\n{loc_list}\nPlease select one by number or name."
            messages.append({"role": "assistant", "content": msg})
            return state.model_copy(update={"messages": messages, "location_options": locations, "location": None, "provider_options": [], "provider": None, "awaiting_confirm": False})

        elif tool_name == "select_location":
            selection = tool_args["selection"]
            selected_loc = _find_selected_location(selection, state.location_options)
            if not selected_loc:
                loc_list = "\n".join([f"{i+1}. {l['name']}" for i, l in enumerate(state.location_options)])
                msg = f"Sorry, I didn't understand that selection. Please pick from the list:\n{loc_list}"
                messages.append({"role": "assistant", "content": msg})
                return state.model_copy(update={"messages": messages})
            msg = f"Great, you've selected {selected_loc['name']}."
            messages.append({"role": "assistant", "content": msg})
            return state.model_copy(update={
                "location": selected_loc["id"],
                "location_name": selected_loc["name"],
                "messages": messages,
                "location_options": [] 
            })

        if tool_name == "get_providers":
            providers = get_providers.invoke(tool_args)
            prov_list = "\n".join([f"{i+1}. {p['name']}" for i, p in enumerate(providers)])
            msg = f"Here are the other available providers at {state.location_name}:\n{prov_list}\nPlease select one by number or name."
            messages.append({"role": "assistant", "content": msg})
            state.awaiting_confirm = False
            return state.model_copy(update={"messages": messages, "provider": None, "provider_name": None, "provider_options": providers})

        elif tool_name == "get_slots":
            state.awaiting_confirm = False
            min_time = tool_args.pop("min_time", None)
            max_time = tool_args.pop("max_time", None)
           
            prov_id = tool_args.get("provider_ids", [state.provider or state.preferred_provider])[0]
            prov_name = state.provider_name or state.preferred_provider_name
            if state.provider_options:
                found_prov = next((p for p in state.provider_options if p['id'] == prov_id), None)
                if found_prov:
                    prov_name = found_prov['name']
            
            tool_args["location_ids"] = [state.location]
            tool_args.setdefault("days", 7)
            
            slots = get_slots.invoke(tool_args)
            if min_time or max_time:
                def get_time(s):
                    return datetime.fromisoformat(s['start_time']).time()
                if min_time:
                    min_t = datetime.strptime(min_time, '%H:%M').time()
                    slots = [s for s in slots if get_time(s) >= min_t]
                if max_time:
                    max_t = datetime.strptime(max_time, '%H:%M').time()
                    slots = [s for s in slots if get_time(s) <= max_t]
            
            if not slots:
                msg = f"I'm sorry, no slots are available for {prov_name} in the specified period. Would you like to change providers or dates?"
                messages.append({"role": "assistant", "content": msg})
                return state.model_copy(update={"messages": messages, "provider": None if "provider" in msg else state.provider, "provider_name": None, "provider_options": []}) 
                
            is_next_available = any(phrase in state.messages[-1]["content"].lower() for phrase in ["next available", "earliest slot", "soonest slot", "first available", "earliest available", "show one slot", "one slot", "the first one"])
            if is_next_available and slots:
                slots = sorted(slots, key=lambda s: s['start_time'])[:1]
                slot_list = _format_slots_for_user(slots)
                msg = f"Here is the next available slot for {prov_name}:\n{slot_list}\nWould you like to book this slot?"
            else:
                slot_list = _format_slots_for_user(slots)
                msg = f"Here are the available slots for {prov_name}:\n{slot_list}\nPlease select a slot by its number."
            messages.append({"role": "assistant", "content": msg})
            return state.model_copy(update={
                "messages": messages, 
                "provider": prov_id, 
                "provider_name": prov_name,
                "provider_options": [], 
                "slot_options": slots
            })

        elif tool_name == "select_slot":
            selection = tool_call["args"]["selection"]
            selected_slot = _find_selected_slot(selection, state.slot_options)
            
            if not selected_slot:
                slot_list = _format_slots_for_user(state.slot_options)
                msg = f"Sorry, I didn't understand that selection. Please pick from the list:\n{slot_list}"
                messages.append({"role": "assistant", "content": msg})
                return state.model_copy(update={"messages": messages})
                
            try:
                time_obj = datetime.fromisoformat(selected_slot['start_time'])
                time_str = time_obj.strftime('%A, %B %d at %I:%M %p')
            except Exception:
                time_str = selected_slot['start_time']
                
            msg = f"Great, I've selected {time_str}."
            messages.append({"role": "assistant", "content": msg})
            return state.model_copy(update={
                "slot_time": selected_slot["start_time"],
                "operatory_id": selected_slot["operatory_id"],
                "messages": messages,
                "slot_options": [] 
            })

    else:
        if not content:
            content = "Could you confirm your above data?"
        messages.append({"role": "assistant", "content": content})

    return state.model_copy(update={"messages": messages})


def new_patient_node(state: AppointmentState) -> AppointmentState:
    if state.slot_time:  
        return state

    messages = state.messages
    system_prompts = []
    tools = []


    if not state.location and not state.location_options:
        system_prompts.append(
            "Your task is to register a new patient. "
            "First, you *must* call `get_locations` to show them the options."
        )
        tools = [get_locations]
    
  
    elif state.location_options and not state.location:
        loc_list_str = "\n".join([f"Number {i+1} ('{l['name']}') is ID {l['id']}" for i, l in enumerate(state.location_options)])
        system_prompts.append(
            "The user was shown a list of locations. They will now select one by name or number. "
            f"You must map their selection to the correct location ID from this list:\n{loc_list_str}\n"
            "Once you have the ID, you *must* call `get_providers(location_id=...)`."
        )
        tools = [get_providers]

   
    elif state.location and state.provider_options and not state.provider:
        prov_list_str = "\n".join([f"Number {i+1} ('{p['name']}') is ID {p['id']}" for i, p in enumerate(state.provider_options)])
        system_prompts.append(
            "The user was shown a list of providers. They will now select one by name or number. "
            f"You must capture their selection as a string (e.g., '1' or 'Dr. Smith'). Here is the list for your reference:\n{prov_list_str}\n"
            "You *must* call `select_provider(selection=...)` with their choice."
        )
        tools = [select_provider]

    
    elif state.provider and not state.email:
            system_prompts.append(
            f"The user has selected provider: {state.provider_name}. "
            "Now, you must ask them for their email address. "
            "Once they provide it, you *must* call `get_email(email=...)`."
        )
            tools = [get_email]

    
    elif state.email and not state.patient_id:
        system_prompts.append(
            f"You have all information to create a new patient:\n"
            f"Name: {state.name}\n"
            f"Phone: {state.phone_number}\n"
            f"DOB: {state.dob}\n"
            f"Email: {state.email}\n"
            f"Location ID: {state.location}\n"
            f"Provider ID: {state.provider}\n"
            "The user has already seen this confirmation. They will now reply 'yes' or 'no'. "
            "If they confirm (e.g., 'yes', 'correct'), you *must* call `create_patient` with all the required arguments: "
            "`provider_id`, `full_name`, `email`, `phone_number`, `date_of_birth`, `location_id`."
        )
        tools = [create_patient]

   
    elif state.patient_id and not state.slot_options and not state.slot_time:
            system_prompts.append(
            "The patient is registered. Ask them for a start date to search for slots. "
            f"Today's date is {datetime.now().strftime('%Y-%m-%d')}. "
            "You MUST convert their answer (e.g.,'today', 'tomorrow', 'next week') into a 'YYYY-MM-DD' `start_date`. "
            "Adjust 'days' based on request (default 1, e.g., 2 for '2 days', 14 for 'two weeks' for today the days are like 1 etc). "
            "If time filters (e.g., 'after 5pm'), include 'min_time': '17:00' and/or 'max_time': 'HH:MM'. "
            "Do NOT ask for the number of days. "
            f"Then, you *must* call `get_slots` with that `start_date`, `days`, `location_ids`=[{state.location}], and `provider_ids`=[{state.provider}]."
        )
            tools = [get_slots]


    elif state.slot_options and not state.slot_time:
        slot_list_str = "\n".join([f"Number {i+1} is '{s['start_time']}' (Operatory: {s['operatory_id']})" for i, s in enumerate(state.slot_options)])
        system_prompts.append(
            "The user was shown a list of slots. They will now select one by number or time. "
            f"You must capture their selection as a string (e.g., '1' or '9am'). Here is the list for your reference:\n{slot_list_str}\n"
            "If the user asks for the first one, earliest, or just one slot, use '1'."
            "You *must* call `select_slot(selection=...)` with their choice."
        )
        tools = [select_slot]
    

    else:
        system_prompts.append("Please guide the user to the next step.")
        tools = [get_locations, get_providers, get_email, create_patient, get_slots, select_slot]



    if messages[-1]["role"] != "system":
        messages.append({"role": "system", "content": "\n".join(system_prompts)})

    response = _get_llm_response(messages, tools)
    content = getattr(response, "content", "")
    tool_calls = getattr(response, "tool_calls", [])


    if messages[-1]["role"] == "system":
        messages.pop()

    if tool_calls:
        tool_call = tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "get_locations":
            locs = get_locations.invoke(tool_args)
            loc_list = "\n".join([f"{i+1}. {l['name']}, {l['city']}" for i, l in enumerate(locs)])
            msg = f"Here are our locations:\n{loc_list}\nPlease select a location by its number or name."
            messages.append({"role": "assistant", "content": msg})
            return state.model_copy(update={"messages": messages, "location_options": locs})

        elif tool_name == "get_providers":
            loc_id = tool_args["location_id"]
     
            loc_name = state.location_name
            if state.location_options:
                found_loc = next((l for l in state.location_options if l['id'] == loc_id), None)
                if found_loc:
                    loc_name = found_loc['name']
            
            provs = get_providers.invoke(tool_args)
            prov_list = "\n".join([f"{i+1}. {p['name']}" for i, p in enumerate(provs)])
            msg = f"Great. Here are the providers at {loc_name}:\n{prov_list}\nPlease select one by number or name."
            messages.append({"role": "assistant", "content": msg})
            return state.model_copy(update={
                "messages": messages, 
                "location": loc_id, 
                "location_name": loc_name,
                "location_options": [], 
                "provider_options": provs
            })
        
        elif tool_name == "select_provider":
            selection = tool_args["selection"]
            selected_provider = _find_selected_provider(selection, state.provider_options)
            
            if not selected_provider:
                prov_list = "\n".join([f"{i+1}. {p['name']}" for i, p in enumerate(state.provider_options)])
                msg = f"Sorry, I didn't understand that selection. Please pick from the list:\n{prov_list}"
                messages.append({"role": "assistant", "content": msg})
                return state.model_copy(update={"messages": messages})

            msg = f"Great, you've selected {selected_provider['name']}. Now, what is your email address?"
            messages.append({"role": "assistant", "content": msg})
            return state.model_copy(update={
                "provider": selected_provider["id"],
                "provider_name": selected_provider["name"],
                "messages": messages,
                "provider_options": [] 
            })

        elif tool_name == "get_email":
            email = tool_args["email"]
            msg = (
                f"Got it. Your email is {email}.\n\n"
                f"Let's confirm your details:\n"
                f"Name: {state.name}\n"
                f"Phone: {state.phone_number}\n"
                f"DOB: {state.dob}\n"
                f"Email: {email}\n"
                f"Location: {state.location_name}\n"
                f"Provider: {state.provider_name}\n\n"
                "Is this correct? (Reply 'yes' or 'no')"
            )
            messages.append({"role": "assistant", "content": msg})
            return state.model_copy(update={"messages": messages, "email": email})

        elif tool_name == "create_patient":
            tool_args["full_name"] = state.name
            tool_args["phone_number"] = state.phone_number
            tool_args["date_of_birth"] = state.dob
            tool_args["email"] = state.email
            tool_args["location_id"] = state.location
            tool_args["provider_id"] = state.provider

            result = create_patient.invoke(tool_args)
            
            if "error" in result and result['error'] is not None:
                msg = f"I'm sorry, there was an error registering you: {result['error']}. Would you like to try again?"
                messages.append({"role": "assistant", "content": msg})
                
                return state.model_copy(update={"messages": messages, "email": None})
            
            patient_id = result.get("patient_id")
            msg = f"Great! You are registered with Patient ID: {patient_id}. Let's find an appointment."
            messages.append({"role": "assistant", "content": msg})
            
            return state.model_copy(update={
                "messages": messages,
                "patient_id": str(patient_id), 
            })

        elif tool_name == "get_slots":
            prov_id = tool_args.get("provider_ids", [state.provider])[0]
            prov_name = state.provider_name
            
            min_time = tool_args.pop("min_time", None)
            max_time = tool_args.pop("max_time", None)
            
            tool_args["location_ids"] = [state.location]
            tool_args.setdefault("days", 7) 

            slots = get_slots.invoke(tool_args)
            if min_time or max_time:
                def get_time(s):
                    return datetime.fromisoformat(s['start_time']).time()
                if min_time:
                    min_t = datetime.strptime(min_time, '%H:%M').time()
                    slots = [s for s in slots if get_time(s) >= min_t]
                if max_time:
                    max_t = datetime.strptime(max_time, '%H:%M').time()
                    slots = [s for s in slots if get_time(s) <= max_t]
            
            if not slots:
                msg = f"I'm sorry, no slots are available for {prov_name} in the next 7 days. Would you like to try again with a different date?"
                messages.append({"role": "assistant", "content": msg})
              
                return state.model_copy(update={"messages": messages, "slot_options": []})

            is_next_available = any(phrase in state.messages[-1]["content"].lower() for phrase in ["next available", "earliest slot", "soonest slot", "first available", "earliest available", "show one slot", "one slot", "the first one"])
            if is_next_available and slots:
                slots = sorted(slots, key=lambda s: s['start_time'])[:1]
                slot_list = _format_slots_for_user(slots)
                msg = f"Here is the next available slot for {prov_name}:\n{slot_list}\nWould you like to book this slot?"
            else:
                slot_list = _format_slots_for_user(slots)
                msg = f"Here are the available slots for {prov_name}:\n{slot_list}\nPlease select a slot by its number."
            messages.append({"role": "assistant", "content": msg})
            return state.model_copy(update={
                "messages": messages, 
                "slot_options": slots
            })

        elif tool_name == "select_slot":
            selection = tool_call["args"]["selection"]
            selected_slot = _find_selected_slot(selection, state.slot_options)
            
            if not selected_slot:
                slot_list = _format_slots_for_user(state.slot_options)
                msg = f"Sorry, I didn't understand that selection. Please pick from the list:\n{slot_list}"
                messages.append({"role": "assistant", "content": msg})
                return state.model_copy(update={"messages": messages})

            try:
                time_obj = datetime.fromisoformat(selected_slot['start_time'])
                time_str = time_obj.strftime('%A, %B %d at %I:%M %p')
            except Exception:
                time_str = selected_slot['start_time']

            msg = f"Great, I've selected {time_str}."
            messages.append({"role": "assistant", "content": msg})
            return state.model_copy(update={
                "slot_time": selected_slot["start_time"],
                "operatory_id": selected_slot["operatory_id"],
                "messages": messages,
                "slot_options": [] 
            })
            
    else:
        if not content:
            content = "Sorry, I had trouble processing that. Can you select a location, provider, or slot?"
        messages.append({"role": "assistant", "content": content})

    return state.model_copy(update={"messages": messages})



def schedule_appointment_node(state: AppointmentState) -> AppointmentState:
    messages = state.messages
    
   
    if any("Please confirm" in m["content"] for m in messages if m["role"] == "assistant"):
       
        system_prompt = "The user is now replying to the confirmation message. They will say 'confirm' or 'cancel'. Use the `user_confirms` tool to capture their 'yes' or 'no' response."
        messages.append({"role": "system", "content": system_prompt})
        response = _get_llm_response(messages, [user_confirms])
        messages.pop() 
    else:
       
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


        conf_input = {
            **ver_input,
            "location_name": state.location_name,
            "provider_name": state.provider_name
        }
        conf_msg = confirmation_node.invoke(conf_input)
        messages.append({"role": "assistant", "content": conf_msg})
        return state.model_copy(update={"messages": messages})

    
    content = getattr(response, "content", "")
    tool_calls = getattr(response, "tool_calls", [])

    if tool_calls and tool_calls[0]["name"] == "user_confirms":
        decision = tool_calls[0]["args"]
        
        if decision["confirmed"]:
            try:
               
                patient_id = state.patient_id
                
                if not patient_id:
                    logging.error(f"CRITICAL: Attempting to book but patient_id is missing. State: {state.is_existing_patient}")
                    raise ValueError("Patient ID is missing, cannot book appointment.")

                appt_payload = {
                    "patient_id": patient_id,
                    "location_id": state.location,
                    "provider_id": state.provider,
                    "operatory_id": state.operatory_id,
                    "start_time": state.slot_time,
                    "appointment_type_id": state.appointment_type_id
                }
                logging.info(f"Making appointment with payload: {appt_payload}")
                
                appt = make_appointment.invoke(appt_payload)

                if "error" in appt and appt['error'] is not None:
                    msg = f"I'm sorry, there was an error while creating appointment: {appt['error']}. Would you like to try again?"
                    messages.append({"role": "assistant", "content": msg})
                    
                    return state.model_copy(update={"messages": messages, "email": None})
                
                msg = f"All set! Your appointment is booked. The ID is: {appt.get('appointment_id', 'N/A')}"
                messages.append({"role": "assistant", "content": msg})
                
            except Exception as e:
                logging.error(f"Appointment booking failed: {e}")
                messages.append({"role": "assistant", "content": f"I'm sorry, something went wrong while booking. Error: {e}"})
            
            return state.model_copy(update={"messages": messages})
        
        else:
            msg = decision.get("message", "Booking canceled. Let me know if you'd like to start over.")
            messages.append({"role": "assistant", "content": msg})
            return state.model_copy(update={"messages": messages})

    if not tool_calls and content:
        messages.append({"role": "assistant", "content": content})
    elif not tool_calls:
        messages.append({"role": "assistant", "content": "Sorry, I didn't understand. Did you want to 'confirm' or 'cancel'?"})
    
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

workflow.add_conditional_edges("patient_info", decide_patient_path, {
    "existing_patient": "existing_patient",
    "new_patient": "new_patient",
    END: END
})

def route_to_schedule(s: AppointmentState):
    """Route to schedule if slot picked; otherwise wait for next user input."""
    if s.slot_time:
        return "schedule_appointment"
    return END

workflow.add_conditional_edges("existing_patient", route_to_schedule, {
    "schedule_appointment": "schedule_appointment",
    END: END
})
workflow.add_conditional_edges("new_patient", route_to_schedule, {
    "schedule_appointment": "schedule_appointment",
    END: END
})

def decide_schedule_path(s: AppointmentState):
    """Decide if booking is done or needs more info."""
    if not s.messages:
        return END
        
    last_msg_content = s.messages[-1]["content"].lower()
    
   
    if "booked" in last_msg_content or "canceled" in last_msg_content:
        return END
        

    if "please confirm" in last_msg_content:
        return END
        
  
    if "missing" in last_msg_content:
        if s.is_existing_patient:
            return "existing_patient"
        return "new_patient"
        
    
    return "schedule_appointment"

workflow.add_conditional_edges("schedule_appointment", decide_schedule_path, {
    "existing_patient": "existing_patient",
    "new_patient": "new_patient",
    "schedule_appointment": "schedule_appointment",
    END: END
})

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
    
        state_input = {**state, "messages": current_messages}

        state = app.invoke(state_input) 
       
        if "messages" in state and state["messages"]:
          
            assistant_messages = [m["content"] for m in state["messages"] if m["role"] == "assistant"]
            if assistant_messages:
                print(f"Assistant: {assistant_messages[-1]}")
     
        if "messages" in state and any("booked" in m["content"].lower() or "canceled" in m["content"].lower() for m in state["messages"] if m["role"] == "assistant"):
            print("\n--- Session ended. Please start a new booking. ---")
            state = {} 
            
if __name__ == "__main__":
    run_interactive_workflow()