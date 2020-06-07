import requests

info = {"text": "How are you doing?"}
response = requests.get("http://localhost:5000/indexabc", data=info)
# print(respons  e.status_code)

# print(response.text)
print(response.json())

