import requests
BASE="http://127.0.0.1:6000/"
response=requests.put(BASE+"category/test",{'lang':'english','trans':'marathi','sentence':'can i help you'})
data=response.json()
print(data)
#response=requests.put(BASE+"category/test",{'lang':'english','trans':'fra','sentence':'can i help you'})
#data=response.json()
#print(data)
#response=requests.put(BASE+"category/test",{'lang':'english','trans':'marathi','sentence':'i am a good person'})
#data=response.json()
#print(data)
#response=requests.put(BASE+"category/test",{'lang':'english','trans':'fra','sentence':'i am a good person'})
#data=response.json()
#print(data)
