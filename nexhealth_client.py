import os
from dotenv import load_dotenv
import requests
import time
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

load_dotenv()

class NexHealthClient:
    """
    Class-based client to manage NexHealth authentication and API calls.
    """
    def __init__(self):
        self.NEXHEALTH_API_KEY = os.getenv('NEXHEALTH_API_KEY')
        self.NEXHEALTH_BASE_URL = os.getenv('NEXHEALTH_BASE_URL')
        self.SUBDOMAIN = os.getenv('NEXHEALTH_SUBDOMAIN')
        
        if not self.NEXHEALTH_BASE_URL or not self.NEXHEALTH_API_KEY or not self.SUBDOMAIN:
            raise EnvironmentError("Missing NexHealth environment variables (NEXHEALTH_API_KEY, NEXHEALTH_BASE_URL, NEXHEALTH_SUBDOMAIN).")
            
        self._token: Optional[str] = None
        self._expires_at: float = 0.0
        try:
            self.authenticate()
            print("NexHealth Client initialized and authenticated.")
        except Exception as e:
            print(f"Failed to initialize NexHealth Client: {e}")

    def _is_token_valid(self) -> bool:
        return bool(self._token) and (self._expires_at > time.time())

    def authenticate(self) -> None:
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
            self._expires_at = time.time() + 3600
        except requests.exceptions.RequestException as e:
            self._token = None
            self._expires_at = 0
            raise ConnectionError(f"Could not authenticate with NexHealth: {e}")

    def get_headers(self) -> dict:
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

    def search_patients(self, name: str, phone_number: str, date_of_birth: str) -> Union[str, List[dict]]:
        """
        Searches for patients matching DOB, then filters by name and phone.
        Does not require a location ID.
        """
        try:
            headers = self.get_headers()
            params = {
                "subdomain": self.SUBDOMAIN,
                "date_of_birth": date_of_birth,
                "location_id": 331668
            }
            endpoint = f"{self.NEXHEALTH_BASE_URL}/patients"
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()
            data = response.json().get("data", {})
            patients = data.get("patients", [])

            print(name, phone_number, date_of_birth)

            if patients: 
                result_patient = []
                for patient in patients:
                    if patient.get('name', '').lower() == name.lower():
                        patient_phone = patient.get('bio', {}).get('phone_number')
                        if patient_phone == phone_number:
                            result_patient.append({
                                'id': patient['id'],
                                'name': patient['name'],
                                'phone': patient_phone,
                                'status': 'verified_patient',
                                'location_ids': patient.get('location_ids', []),
                                'upcoming_appts': patient.get('upcoming_appts', []),
                                'appointments': patient.get('appointments', [])
                            })
                        else:
                            result_patient.append({
                                'id': patient['id'],
                                'name': patient['name'],
                                'phone': patient_phone,
                                'status': 'phone_number_mismatch',
                                'location_ids': patient.get('location_ids', []),
                                'upcoming_appts': patient.get('upcoming_appts', [])
                            })
                
                if result_patient:
                    return result_patient
            
            return "no_patient_found"
        except Exception as e:
            return f"Error fetching patient information: {e}"