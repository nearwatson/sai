import requests

info = {"text": "How are you doing?"}
response = requests.get("http://localhost:5000", data=info)
# print(response.status_code)

# print(response.text)
print(response.json())