from datetime import datetime, timedelta
import requests
import os
from dotenv import load_dotenv
import pandas as pd
import time

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise EnvironmentError("API_KEY not found in environment variables. Check your .env file.")

# Coordinates and date range
LAT, LON = 27.8, 67.9
end_time = int(datetime.now().timestamp())
start_time = int((datetime.now() - timedelta(days=90)).timestamp())
print(f"Fetching data from {datetime.fromtimestamp(start_time)} to {datetime.fromtimestamp(end_time)}")

# Reusable function to fetch data in chunks
def fetch_data(url_template, start, end, chunk_days=5, data_parser=None, label="data"):
    all_data = []
    current = start

    while current < end:
        next_chunk = min(current + chunk_days * 86400, end)
        url = url_template.format(start=current, end=next_chunk, lat=LAT, lon=LON, key=API_KEY)
        response = requests.get(url)

        if response.ok:
            raw_data = response.json().get("list", [])
            parsed = data_parser(raw_data) if data_parser else raw_data
            all_data.extend(parsed)
            if parsed:
                print(f"Got {label} from {datetime.fromtimestamp(parsed[0]['timestamp'])} to {datetime.fromtimestamp(parsed[-1]['timestamp'])}")
        else:
            print(f"Failed to fetch {label} from {datetime.fromtimestamp(current)} to {datetime.fromtimestamp(next_chunk)}")
            print(f"Status code: {response.status_code}, Response: {response.text}")

        current = next_chunk
        time.sleep(1)

    return all_data

# Pollution data parser
def parse_pollution_data(data):
    return [
        {
            "timestamp": item["dt"],
            "aqi": item["main"]["aqi"],
            "pm2_5": round(item["components"]["pm2_5"], 2),
            "pm10": round(item["components"]["pm10"], 2),
            "co": round(item["components"]["co"], 2),
        }
        for item in data
    ]

# Weather data parser
def parse_weather_data(data):
    return [
        {
            "timestamp": item["dt"],
            "temp": round(item["main"]["temp"], 2),
            "humidity": round(item["main"]["humidity"], 2),
        }
        for item in data
    ]

# Fetch both datasets
pollution_url = "http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}&appid={key}"
weather_url = "https://history.openweathermap.org/data/2.5/history/city?lat={lat}&lon={lon}&type=hour&start={start}&end={end}&units=metric&appid={key}"

print("Fetching pollution data...")
pollution_data = fetch_data(pollution_url, start_time, end_time, data_parser=parse_pollution_data, label="pollution")

print("Fetching weather data...")
weather_data = fetch_data(weather_url, start_time, end_time, data_parser=parse_weather_data, label="weather")

# Process pollution data
df_pollution = pd.DataFrame(pollution_data)
if not df_pollution.empty:
    df_pollution['date'] = pd.to_datetime(df_pollution['timestamp'], unit='s').dt.date
    daily_pollution = df_pollution.groupby('date').mean(numeric_only=True).round(2).reset_index()
    print(f"Pollution data date range: {daily_pollution['date'].min()} to {daily_pollution['date'].max()}")
else:
    print("No pollution data found.")
    daily_pollution = pd.DataFrame(columns=['date', 'aqi', 'pm2_5', 'pm10', 'co'])

# Process weather data
df_weather = pd.DataFrame(weather_data)
if not df_weather.empty:
    df_weather['date'] = pd.to_datetime(df_weather['timestamp'], unit='s').dt.date
    agg_funcs = {
        'temp': ['min', 'max'],
        'humidity': ['min', 'max']
    }
    daily_weather = df_weather.groupby('date').agg(agg_funcs)
    daily_weather.columns = ['temp_low', 'temp_high', 'humidity_low', 'humidity_high']
    daily_weather = daily_weather.round(2).reset_index()
    print(f"Weather data date range: {daily_weather['date'].min()} to {daily_weather['date'].max()}")
else:
    print("No weather data found.")
    daily_weather = pd.DataFrame(columns=['date', 'temp_low', 'temp_high', 'humidity_low', 'humidity_high'])

# Merge and save final data
if not daily_pollution.empty and not daily_weather.empty:
    merged_df = pd.merge(daily_pollution, daily_weather, on='date', how='inner', validate='one_to_one')
    print(f"Merged data date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    merged_df.to_csv("daily_aqi_weather_90days.csv", index=False)
    print("Saved merged data to daily_aqi_weather_90days.csv")
else:
    print("Merging skipped. One or both dataframes are empty.")
