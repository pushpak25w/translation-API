import requests
BASE="http://127.0.0.1:6000/"
response=requests.put(BASE+"category/test",{'lang':'english','trans':'marathi','sentence':'give it to me'})
data=response.json()
print(data)
