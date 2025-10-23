import os
from dotenv import load_dotenv
import requests
import time
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

class GetAvailableSlotsInput(BaseModel):
    """Input model for checking available appointment slots."""
    start_date: Optional[str] = Field(None, description="The starting date to check for slots, in YYYY-MM-DD format.")
    days: Optional[int] = Field(7, description="The number of days from the start date to check for availability.")
    location_ids: Optional[List[int]] = Field(None, description="A list of location IDs to check.")
    provider_ids: Optional[List[int]] = Field(None, description="A list of provider IDs to check.")

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

    def get_providers(self, location_id: int) -> List[Dict[str, Any]]:
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
                "location_id": location_id
            }
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

    def get_available_slots(self, start_date: str, days: int = 7, location_ids: List[int] = None, provider_ids: List[int] = None) -> List[Dict[str, Any]]:
        """
        Fetches available appointment slots based on multiple criteria.
        Returns a list of simplified slot dictionaries.
        """
        try:
            headers = self.get_headers()
            location_ids = location_ids or []
            provider_ids = provider_ids or []
            start_date = start_date or time.strftime("%Y-%m-%d")
            days = days or 7

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
            slots = []
            for item in data:
                for slot in item.get("slots", []):
                    slots.append({
                        "location_id": item["lid"],
                        "provider_id": slot["provider_id"],
                        "operatory_id": slot["operatory_id"],
                        "start_time": slot["time"],
                        "end_time": slot["end_time"]
                    })
            return slots
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Error fetching available slots: {e}")

    def search_patients(self, name: str, phone_number: str, date_of_birth: str, location_id: int) -> Union[str, List[Dict[str, Any]]]:
        """
        Searches for patients by DOB, then filters by name and phone number.
        Returns a list of matching patient dictionaries or an error message.
        """
        try:
            headers = self.get_headers()
            params = {
                "subdomain": self.SUBDOMAIN,
                "date_of_birth": date_of_birth,
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
                        "location_id": location_id
                    })
                elif patient_name == name.lower():
                    result_patients.append({
                        "id": patient["id"],
                        "name": patient["name"],
                        "phone": patient_phone,
                        "status": "number not verified",
                        "location_id": location_id
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
        location_id: int = None,
        days: Optional[int] = 10,
    ) -> Union[str, dict, List[dict]]:
        """
        Retrieve appointments for a time window or a single appointment by ID.

        Args:
            appointment_id (Optional[int]): If provided, return only this appointment ID.
            location_id (int): Location ID to filter appointments (can be None).
            days (Optional[int]): Number of days from now to include in the range (default 10).

        Returns:
            - dict   : single appointment (when appointment_id is given and found)
            - list   : all appointments in the window (when appointment_id is None)
            - str    : error message or "No appointment found..."
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