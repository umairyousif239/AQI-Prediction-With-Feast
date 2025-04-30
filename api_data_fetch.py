from datetime import datetime, timedelta, timezone
import requests
import os
from dotenv import load_dotenv
import pandas as pd
import time
from pathlib import Path
import pytz

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise EnvironmentError("API_KEY not found in environment variables. Check your .env file.")

# Coordinates and timezone
LAT, LON = 27.8, 67.9 # Location: Shahdadkot
my_timezone = timezone.utc

# Set file path
file_path = "feature_repo/data/daily_aqi_weather_90days.parquet"

# Check if existing data exists
if Path(file_path).exists():
    existing_df = pd.read_parquet(file_path)
    print(f"Loaded existing data with {len(existing_df)} rows.")
    
    # Find the latest date in existing data
    latest_date = pd.to_datetime(existing_df["date_key"]).max()

    # Define timezone-aware "yesterday" timestamp
    now = datetime.now(my_timezone)
    yesterday = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    # Calculate next start time from latest_date + 1 day
    candidate_start = (latest_date + timedelta(days=1)).replace(tzinfo=my_timezone)

    # Set start_time as the earlier of candidate_start and yesterday
    final_start = min(candidate_start, yesterday)
    start_time = int(final_start.timestamp())
else:
    existing_df = None
    print("No existing data found. Starting fresh.")
    
    # If no data, fetch last 90 days
    start_time = int((datetime.now(my_timezone) - timedelta(days=90)).timestamp())

# End time is now
end_time = int(datetime.now(my_timezone).timestamp())

print(f"Fetching data from {datetime.fromtimestamp(start_time, my_timezone)} to {datetime.fromtimestamp(end_time, my_timezone)}")

# Fetch data in chunks
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

# Parse functions
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

def parse_weather_data(data):
    return [
        {
            "timestamp": item["dt"],
            "temp": round(item["main"]["temp"], 2),
            "humidity": round(item["main"]["humidity"], 2),
        }
        for item in data
    ]

# Parse current weather data - different format from historical data
def parse_current_weather(data):
    return [{
        "timestamp": data["dt"],
        "temp": round(data["main"]["temp"], 2),
        "humidity": round(data["main"]["humidity"], 2),
    }]

# Parse current pollution data
def parse_current_pollution(data):
    if "list" in data and data["list"]:
        item = data["list"][0]  # Get the first (and only) item
        return [{
            "timestamp": item["dt"],
            "aqi": item["main"]["aqi"],
            "pm2_5": round(item["components"]["pm2_5"], 2),
            "pm10": round(item["components"]["pm10"], 2),
            "co": round(item["components"]["co"], 2),
        }]
    return []

# URLs
pollution_url = "http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}&appid={key}"
weather_url = "https://history.openweathermap.org/data/2.5/history/city?lat={lat}&lon={lon}&type=hour&start={start}&end={end}&units=metric&appid={key}"

# API for today's data
today_pollution_url = "http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={key}"
today_weather_url = "https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={key}"

# Fetch historical data
print("Fetching historical pollution data...")
pollution_data = fetch_data(pollution_url, start_time, end_time, data_parser=parse_pollution_data, label="pollution")

print("Fetching historical weather data...")
weather_data = fetch_data(weather_url, start_time, end_time, data_parser=parse_weather_data, label="weather")

# Fetch today's current data
print("Fetching today's current pollution data...")
today_pollution_response = requests.get(today_pollution_url.format(lat=LAT, lon=LON, key=API_KEY))
if today_pollution_response.ok:
    current_pollution = parse_current_pollution(today_pollution_response.json())
    if current_pollution:
        print(f"Got current pollution data for {datetime.fromtimestamp(current_pollution[0]['timestamp'])}")
        pollution_data.extend(current_pollution)
else:
    print(f"Failed to fetch current pollution. Status: {today_pollution_response.status_code}")

print("Fetching today's current weather data...")
today_weather_response = requests.get(today_weather_url.format(lat=LAT, lon=LON, key=API_KEY))
if today_weather_response.ok:
    current_weather = parse_current_weather(today_weather_response.json())
    if current_weather:
        print(f"Got current weather data for {datetime.fromtimestamp(current_weather[0]['timestamp'])}")
        weather_data.extend(current_weather)
else:
    print(f"Failed to fetch current weather. Status: {today_weather_response.status_code}")

# Process pollution data
df_pollution = pd.DataFrame(pollution_data)
if not df_pollution.empty:
    df_pollution["date"] = pd.to_datetime(df_pollution["timestamp"], unit="s").dt.date
    df_pollution = df_pollution.drop(columns=["timestamp"])
    daily_pollution = df_pollution.groupby("date").mean(numeric_only=True).round(2).reset_index()
    print(f"Pollution data date range: {daily_pollution['date'].min()} to {daily_pollution['date'].max()}")
else:
    print("No pollution data found.")
    daily_pollution = pd.DataFrame(columns=["date", "aqi", "pm2_5", "pm10", "co"])

# Process weather data
df_weather = pd.DataFrame(weather_data)
if not df_weather.empty:
    df_weather["date"] = pd.to_datetime(df_weather["timestamp"], unit="s").dt.date
    df_weather = df_weather.drop(columns=["timestamp"])
    daily_weather = df_weather.groupby("date").agg({
        "temp": ["min", "max"],
        "humidity": ["min", "max"]
    })
    daily_weather.columns = ["temp_low", "temp_high", "humidity_low", "humidity_high"]
    daily_weather = daily_weather.round(2).reset_index()
    print(f"Weather data date range: {daily_weather['date'].min()} to {daily_weather['date'].max()}")
else:
    print("No weather data found.")
    daily_weather = pd.DataFrame(columns=["date", "temp_low", "temp_high", "humidity_low", "humidity_high"])

# Merge and save
if not daily_pollution.empty and not daily_weather.empty:
    merged_df = pd.merge(daily_pollution, daily_weather, on="date", how="inner", validate="one_to_one")
    print(f"Merged data date range: {merged_df['date'].min()} to {merged_df['date'].max()}")

    # Convert 'date' to 'date_key' and 'timestamp_key'
    merged_df["date_key"] = pd.to_datetime(merged_df["date"]).dt.strftime('%Y-%m-%d')
    merged_df["timestamp_key"] = pd.to_datetime(merged_df["date_key"])
    merged_df = merged_df.drop(columns=["date"])

    # Reorder columns
    cols = merged_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("date_key")))
    cols.insert(1, cols.pop(cols.index("timestamp_key")))
    merged_df = merged_df[cols]

    # Merge with existing data if any
    if existing_df is not None:
        combined_df = pd.concat([existing_df, merged_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["date_key"], keep="last").sort_values("date_key").reset_index(drop=True)
    else:
        combined_df = merged_df

    # Backup the existing Parquet file before overwriting
    if Path(file_path).exists():
        backup_path = f"{file_path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
        Path(file_path).rename(backup_path)
        print(f"Backup created at {backup_path}")

    # Save updated data
    combined_df.to_parquet(file_path, index=False)
    print(f"Saved updated data with {len(combined_df)} rows to {file_path}")


else:
    print("Merging skipped. One or both dataframes are empty.")