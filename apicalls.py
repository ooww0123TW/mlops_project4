import subprocess
import json
import os

with open('config.json','r') as f:
    config = json.load(f) 

output_model_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"


#Call each API endpoint and store the responses
response1 = subprocess.run(['curl', '-X', 'POST', URL + 'prediction?data=' + test_data_path], capture_output=True).stdout
response2 = subprocess.run(['curl', URL + 'scoring'], capture_output=True).stdout
response3 = subprocess.run(['curl', URL + 'summarystats'], capture_output=True).stdout
response4 = subprocess.run(['curl', URL + 'diagnostics'], capture_output=True).stdout


#combine all API responses
responses = b"".join([response1, response2, response3, response4])

# write the responses to your workspace
apireturns_path = os.path.join(output_model_path, 'apireturns2.txt')
with open(apireturns_path, "w") as file:
    file.write(responses.decode("utf-8"))