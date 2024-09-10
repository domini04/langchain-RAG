import requests
import uuid

# Generate a unique session ID for this interaction (can reuse across requests for persistent session)
# session_id = str(uuid.uuid4())
session_id = str(123)

# Define the API URL
url = "http://localhost:8000/ask"

# The input data for the API request
data = {
    "question": "내 이름은 엽이라고 해",  # Example question
    "session_id": session_id  # Use the generated session ID
}

# Send the POST request
response = requests.post(
    url=url,
    json=data
)

# Decode the response content to utf-8 and print
print(response.content.decode('utf-8'))
