import requests
import os
from dotenv import load_dotenv

load_dotenv()
api = os.getenv("open_weather_api")
print(api)