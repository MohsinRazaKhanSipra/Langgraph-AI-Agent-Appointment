import os
from dotenv import load_dotenv
import requests
import time
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

class NexHealthClient:
    """
    Class-based client to manage NexHealth authentication and API calls.
    Use NexHealthClient().get_headers() to retrieve a valid Authorization header.
    """
    NEXHEALTH_API_KEY = os.getenv('NEXHEALTH_API_KEY')
    NEXHEALTH_BASE_URL = os.getenv('NEXHEALTH_BASE_URL')
    SUBDOMAIN = os.getenv('NEXHEALTH_SUBDOMAIN')

    def __init__(self):
        self._token: Optional[str] = None
        self._expires_at: float = 0.0
        self.authenticate()

        if not self.NEXHEALTH_BASE_URL or not self.NEXHEALTH_API_KEY:
            raise EnvironmentError("Missing NexHealth base URL or API key.")

    def _is_token_valid(self) -> bool:
        return bool(self._token) and (self._expires_at > time.time())

    def authenticate(self) -> None:
        """
        Authenticate against NexHealth and populate internal token cache.
        Raises ConnectionError on network/auth failure.
        """
        auth_url = f"{self.NEXHEALTH_BASE_URL}/authenticates"
        auth_headers = {
            "accept": "application/vnd.Nexhealth+json;version=2",
            "Authorization": self.NEXHEALTH_API_KEY
        }

        try:
            resp = requests.post(auth_url, headers=auth_headers)
            resp.raise_for_status()
            data = resp.json()
            bearer_token = data.get("data", {}).get("token")
            if not bearer_token:
                self._token = None
                self._expires_at = 0
                raise ValueError("Failed to retrieve bearer token from NexHealth.")
            self._token = bearer_token
            self._expires_at = time.time() + 3600  # 1 hour expiry
        except requests.exceptions.RequestException as e:
            self._token = None
            self._expires_at = 0
            raise ConnectionError(f"Could not authenticate with NexHealth: {e}")

    def get_headers(self) -> dict:
        """
        Returns headers with a valid Bearer token, authenticating if necessary.
        """
        if self._is_token_valid():
            return {
                "accept": "application/vnd.Nexhealth+json;version=2",
                "Authorization": f"Bearer {self._token}"
            }
        self.authenticate()
        return {
            "accept": "application/vnd.Nexhealth+json;version=2",
            "Authorization": f"Bearer {self._token}"
        }

    def get_locations(self) -> List[Dict[str, Any]]:
        """
        Fetches all locations from the NexHealth API and formats them for display.
        Returns a list of simplified location dictionaries.
        """
        try:
            headers = self.get_headers()
            response = requests.get(f"{self.NEXHEALTH_BASE_URL}/locations", headers=headers)
            response.raise_for_status()
            data = response.json().get("data", [])
            locations = []
            for item in data:
                for loc in item.get("locations", []):
                    locations.append({
                        "id": loc["id"],
                        "name": loc["name"],
                        "street_address": loc["street_address"],
                        "city": loc["city"],
                        "state": loc["state"],
                        "zip_code": loc["zip_code"],
                        "phone_number": loc["phone_number"],
                        "email": loc["email"],
                        "timezone": loc["tz"]
                    })
            return locations
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Error fetching locations: {e}")

    def get_providers(self, location_id: int=None) -> List[Dict[str, Any]]:
        """
        Fetches providers for the configured subdomain and specified location.
        Returns a list of simplified provider dictionaries.
        """
        try:
            headers = self.get_headers()
            params = {
                "subdomain": self.SUBDOMAIN,
                "inactive": "false",
                "include[]": "appointment_types",
            }

            if location_id is not None:
                params["location_id"] = location_id

            response = requests.get(f"{self.NEXHEALTH_BASE_URL}/providers", headers=headers, params=params)
            response.raise_for_status()
            data = response.json().get("data", [])
            providers = []
            for provider in data:
                providers.append({
                    "id": provider["id"],
                    "name": provider["name"],
                    "display_name": provider["display_name"],
                    "specialty_code": provider.get("specialty_code", ""),
                    "locations": [loc["id"] for loc in provider.get("locations", [])],
                    "appointment_types": [
                        {"id": apt["id"], "name": apt["name"], "minutes": apt["minutes"]}
                        for apt in provider.get("availabilities", [])[0].get("appointment_types", [])
                    ]
                })
            return providers
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Error fetching providers: {e}")

    def get_available_slots(self, start_date: str = None, days: int=None, location_ids: List[int] = None, provider_ids: List[int] = None) -> List[Dict[str, Any]]:
        """
        Fetches available appointment slots based on multiple criteria.
        Returns a list of simplified slot dictionaries.
        

        """
        try:
            headers = self.get_headers()
      


            if location_ids is not None:
                location_ids = location_ids
            else:
                raw_locations = self.get_locations()
                location_ids = [
                loc['id']
                for loc in (raw_locations if isinstance(raw_locations, list) else [])
                if isinstance(loc, dict) and 'id' in loc
                ]

            if provider_ids is not None:
                provider_ids = provider_ids
            else:
                raw_providers = self.get_providers()
                provider_ids = [
                prov['id']
                for prov in (raw_providers if isinstance(raw_providers, list) else [])
                if isinstance(prov, dict) and 'id' in prov
                ]
            start_date = start_date if start_date is not None else time.strftime("%Y-%m-%d")
            days = days if days is not None else 7

            params = {
                "subdomain": self.SUBDOMAIN,
                "start_date": start_date,
                "days": days,
                "lids[]": location_ids,
                "pids[]": provider_ids
            }
            response = requests.get(f"{self.NEXHEALTH_BASE_URL}/appointment_slots", headers=headers, params=params)
            response.raise_for_status()
            data = response.json().get("data", [])
            results = []
            for item in data:
                pid = item.get('pid')
                lid = item.get('lid')
                slots = item.get('slots', [])
                if slots:
                    for slot in slots:
                        results.append(
                            {
                                "provider_id": pid,
                                "location_id": lid,
                                "start_time": slot.get('time'),
                                "end_time": slot.get('end_time'),
                                "operatory_id": slot.get('operatory_id')
                            }
                            
                        )
            return results
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Error fetching available slots: {e}")

    def search_patients(self, name: str, phone_number: str, dob: str, location_id: int) -> Union[str, List[Dict[str, Any]]]:
        """
        Searches for patients by DOB, then filters by name and phone number.
        Returns a list of matching patient dictionaries or an error message.
        """
        try:
            headers = self.get_headers()
            params = {
                "subdomain": self.SUBDOMAIN,
                "date_of_birth": dob, # CHANGED from date_of_birth
                "location_id": location_id
            }
            endpoint = f"{self.NEXHEALTH_BASE_URL}/patients"
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()
            data = response.json().get("data", {})
            patients = data.get("patients", [])

            result_patients = []
            for patient in patients:
                patient_name = patient.get("name", "").lower()
                patient_phone = patient.get("bio", {}).get("phone_number", "")
                
                if patient_name == name.lower() and patient_phone == phone_number:
                    result_patients.append({
                        "id": patient["id"],
                        "name": patient["name"],
                        "phone": patient_phone,
                        "status": "verified patient",
                        "location_ids": [location_id], # CHANGED from location_id
                        "upcoming_appts": patient.get("upcoming_appts", []), # ADDED
                        "provider_id": patient.get("provider_id") # ADDED
                    })
                elif patient_name == name.lower():
                    result_patients.append({
                        "id": patient["id"],
                        "name": patient["name"],
                        "phone": patient_phone,
                        "status": "number not verified",
                        "location_ids": [location_id], # CHANGED from location_id
                        "upcoming_appts": patient.get("upcoming_appts", []), # ADDED
                        "provider_id": patient.get("provider_id") # ADDED
                    })

            return result_patients if result_patients else "no patient found"
        except requests.exceptions.RequestException as e:
            return f"Error fetching patient information: {e}"

    def make_appointment(self, patient_id: str, location_id: int, provider_id: int, operatory_id: int, start_time: str, appointment_type_id: int) -> Dict[str, Any]:
        """
        Creates an appointment in the NexHealth system.
        Returns the created appointment details or raises an error.
        """
        try:
            # headers = self.get_headers()
            # payload = {
            #     "subdomain": self.SUBDOMAIN,
            #     "patient_id": patient_id,
            #     "location_id": location_id,
            #     "provider_id": provider_id,
            #     "operatory_id": operatory_id,
            #     "start_time": start_time,
            #     "appointment_type_id": appointment_type_id
            # }

            # MOCK RESPONSE (as in original file)
            return {
                "appointment_id": 0,
                "patient_id": patient_id,
                "location_id": location_id,
                "provider_id": provider_id,
                "operatory_id": operatory_id,
                "start_time": start_time,
                "status": "confirmed"
            }
        
            # response = requests.post(f"{self.NEXHEALTH_BASE_URL}/appointments", headers=headers, json=payload)
            # response.raise_for_status()
            # data = response.json().get("data", {})
            # return {
            #     "appointment_id": data.get("id"),
            #     "patient_id": data.get("patient_id"),
            #     "location_id": data.get("location_id"),
            #     "provider_id": data.get("provider_id"),
            #     "operatory_id": data.get("operatory_id"),
            #     "start_time": data.get("start_time"),
            #     "status": "confirmed"
            # }
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Error creating appointment: {e}")
    
    def view_appointment(
        self,
        appointment_id: Optional[int] = None,
        location_id: Optional[int] = None, 
        days: Optional[int] = 10,
    ) -> Union[str, dict, List[dict]]:
        """
        Retrieve appointments for a time window or a single appointment by ID.
        """
        if appointment_id is not None and not isinstance(appointment_id, int):
            return "appointment_id must be an integer"


        now = datetime.utcnow()
        start = now.strftime("%Y-%m-%dT%H:%M:%S+0000")
        end_dt = now + timedelta(days=days)
        end = end_dt.strftime("%Y-%m-%dT%H:%M:%S+0000")


        params = {
            "subdomain": self.SUBDOMAIN,
            "start": start,
            "end": end,
        }
        if location_id is not None:
            params["location_id"] = location_id

        url = f"{self.NEXHEALTH_BASE_URL}/appointments"

        try:
            headers = self.get_headers()
            resp = requests.get(url, headers=headers, params=params)
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data", [])

            if appointment_id is not None:
                for appt in data:
                    if appt.get("id") == appointment_id:
                        return appt
                return f"No appointment found with id {appointment_id} in the requested range."

            return data

        except requests.exceptions.RequestException as e:
            return f"Error fetching appointments: {e}"
        except Exception as e:
            return f"Unexpected error fetching appointments: {e}"


    def create_patient(
        self,
        provider_id: int,
        first_name: str,
        last_name: str,
        email: str,
        phone_number: str,
        date_of_birth: str,
        location_id: int
    ) -> Dict[str, Any]:
        """
        Creates a new patient in the NexHealth system.
        Returns simplified patient details or raises an error.
        """
        try:
            headers = self.get_headers()
            headers["content-type"] = "application/json"
            
            payload = {
                "provider": {"provider_id": provider_id},
                "patient": {
                    "bio": {
                        "phone_number": phone_number,
                        "date_of_birth": date_of_birth
                    },
                    "email": email,
                    "last_name": last_name,
                    "first_name": first_name
                }
            }
            
            url = f"{self.NEXHEALTH_BASE_URL}/patients?subdomain={self.SUBDOMAIN}&location_id={location_id}"
            
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json().get("data", {}).get("user", {})
            
            return {
                "id": data.get("id"),
                "email": data.get("email"),
                "first_name": data.get("first_name"),
                "last_name": data.get("last_name"),
                "name": data.get("name"),
                "phone_number": data.get("bio", {}).get("phone_number"),
                "date_of_birth": data.get("bio", {}).get("date_of_birth"),
                "created_at": data.get("created_at"),
                "location_ids": data.get("location_ids", [])
            }
        
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Error creating patient: {e}")
        



# client=NexHealthClient()
# result=client.search_patients("Achaias Tyrell","4692696088","1983-01-31", 331668)
# print("------------ Test 1 -----------")
# print('("Achaias Tyrell","4692696088","1983-01-31", 331668)')
# print(result)
# print("\n\n")

# #Appointment related question prompts
#         #name
#         #dob
#         #telephone



# print("------------ Test 2 -----------")
# print('("ACHAIAS Tyrell","4692696088","1983-01-31", 331668)')
# result=client.search_patients("ACHAIAS Tyrell","4692696088","1983-01-31", 331668)
# print(result)
# print("\n\n")

# print("------------ Test 3 -----------")
# print('("Achaias Tyrell","3692696088","1983-01-31", 331668)')
# result=client.search_patients("Achaias Tyrell","3692696088","1983-01-31", 331668)
# print(result)
# print("\n\n")


# print("------------ Test 4 -----------")
# print('("Achaias Tyrell","4692696088","2000-01-30", 331668)')
# result=client.search_patients("Achaias Tyrell","4692696088","2000-01-30", 331668)
# print(result)
# print("\n\n")



# print("------------ Test 5 -----------")
# print('("Achaias Tyrell","4692696088","2000-01-30", 331668)')
# result=client.search_patients("Mohsin Tyrell","4692696088","1983-01-31", 331668)
# print(result)
# print("\n\n")






# print("------------ get_available_slots() -----------")
# # input_data=[
# #             # start_date="2025-10-16",
# #             # days=1,
# #             # location_ids=[331668],
# #             # provider_ids=[413326781]
# #         ]
# result=client.get_available_slots()
# print(result)
# print("\n\n")



# print("------------- create_patient() --------------")
# result=client.create_patient(
#     provider_id=413326781,
#     first_name="First343Name",
#     last_name="Last343Name",
#     email="example@example1.com",
#     phone_number="3392696088",
#     date_of_birth="1990-01-01",
#     location_id=331668
# )
# print(result)


# print(client.view_appointment(1036290875))

# print(client.view_appointment(1029645607, location_id=331668))

# print(client.view_appointment(location_id=331668))


# print(client.view_appointment(location_id=331668, days=30))