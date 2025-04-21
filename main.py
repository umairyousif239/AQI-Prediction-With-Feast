import datetime as dt
import requests
import os
from dotenv import load_dotenv

load_dotenv()
AQI_API = os.getenv("AQI_API")
Weather_API = os.getenv("Weather_API")

response = requests.get(AQI_API).json()
response2 = requests.get(Weather_API).json()

print(response)
print(response2)