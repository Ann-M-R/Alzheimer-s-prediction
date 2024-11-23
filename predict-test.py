# %%
import requests

# %%
url = 'http://localhost:9696/predict'

# %%
data = {
    "age": 67,
    "gender": 1,
    "ethnicity": "caucasian",
    "educationlevel": "high_school",
    "bmi": 19.364545083578612,
    "smoking": 1,
    "alcoholconsumption": 18.23714054886429,
    "physicalactivity": 1.4770531540776022,
    "dietquality": 3.4808279159120548,
    "sleepquality": 7.471877693572239,
    "familyhistoryalzheimers": 0,
    "cardiovasculardisease": 0,
    "diabetes": 0,
    "depression": 0,
    "headinjury": 0,
    "hypertension": 0,
    "systolicbp": 166,
    "diastolicbp": 93,
    "cholesteroltotal": 295.9802426390147,
    "cholesterolldl": 115.01580550387871,
    "cholesterolhdl": 82.49633979467387,
    "cholesteroltriglycerides": 195.39322332666708,
    "mmse": 22.866157399945575,
    "functionalassessment": 1.8043918926760538,
    "memorycomplaints": 0,
    "behavioralproblems": 0,
    "adl": 8.8256852341999,
    "confusion": 0,
    "disorientation": 0,
    "personalitychanges": 0,
    "difficultycompletingtasks": 0,
    "forgetfulness": 0
}



# %%
requests.post(url, json = data)

# %%
response = requests.post(url, json = data).json()

print(response)

if response['diagnosis'] == True:
    print("The patient is daignosed with alzheimer's")
else:
    print("THe patient does not have alzheimer's")