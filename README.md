# Face-Recognition-API
## Use this Simple Face-Recognition API in your projects Just only by making post requests

using Python 3

1. Add New User

r = requests.post(URL+'/add_user',{"username":"Anshul Nagar","password":"12345"})
r.text

2. Add a Face

r = requests.post(URL+'/add',{"name":"Anna","image":{% DataURI of Image %},"username":"Anshul Nagar","password":"12345"})
r.text

3. Remove a Face

r = requests.post(URL+'/remove',{"name":"Anna","username":"Anshul Nagar","password":"12345"})
r.text

4. Using Face Detection

r = requests.post(URL+'/add',{"image":{% DataURI of Image %},"username":"Anshul Nagar","password":"12345"})
r.text
