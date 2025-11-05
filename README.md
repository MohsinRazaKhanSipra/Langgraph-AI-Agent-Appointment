# AI Appointment Scheduling Agent

An intelligent appointment scheduling system built with LangGraph and the NexHealth API. This agent helps manage medical appointments through natural language conversation, handling both new and existing patients.


Existing test patient:  Name: "Achaias Tyrell",
                        Phone: "4692696088"
                        DOB: "1983-01-31"

## Features

- 🤖 Natural language conversation for appointment scheduling
- 👥 Patient management (new registration and existing patient lookup)
- 📍 Multiple location support
- 👨‍⚕️ Provider selection and management
- 📅 Smart slot availability search and booking
- ✅ Appointment confirmation and verification
- 🔄 Error handling and recovery

## System Requirements

- Python 3.8+
- OpenRouter API access
- NexHealth API access

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/Langgraph-AI-Agent-Appointment.git
cd Langgraph-AI-Agent-Appointment
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root with the following variables:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
NEXHEALTH_API_KEY=your_nexhealth_api_key
NEXHEALTH_BASE_URL=your_nexhealth_base_url
NEXHEALTH_SUBDOMAIN=your_nexhealth_subdomain
```

## Project Structure

```
├── app.py              # Main application with LangGraph agent logic
├── nexhealth_client.py # NexHealth API client implementation
├── requirements.txt    # Project dependencies
└── .env               # Environment variables (create this file)
```

## Usage

1. Ensure all environment variables are properly set in `.env`
2. Run the application:
```bash
python app.py
```

## Agent Workflow

1. **Patient Information Collection**
   - Collects name, phone number, and date of birth
   - Verifies existing patient status

2. **Patient Registration (New Patients)**
   - Location selection
   - Provider selection
   - Email collection
   - Patient record creation

3. **Appointment Details**
   - Location confirmation
   - Provider selection/confirmation
   - Slot availability search
   - Time slot selection

4. **Appointment Scheduling**
   - Data verification
   - Final confirmation
   - Appointment creation

## State Management

The agent maintains state using the `AppointmentState` class, which tracks:
- Patient information
- Location and provider details
- Appointment slot details
- Conversation history
- Available options for selection

## Error Handling

The system includes robust error handling for:
- Invalid patient information
- Unavailable time slots
- API connection issues
- Invalid selections
- Incomplete information

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.