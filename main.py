from datetime import datetime, timedelta
import requests
import os
from dotenv import load_dotenv
import pandas as pd
import time

load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    print("Error: API_KEY not found in environment variables. Check your .env file.")
    exit(1)

LAT, LON = 27.8, 67.9

# Calculate the correct date range - from 90 days ago to today
current_date = datetime.now()
end_time = int(current_date.timestamp())
start_time = int((current_date - timedelta(days=90)).timestamp())

print(f"Requesting data from {datetime.fromtimestamp(start_time)} to {datetime.fromtimestamp(end_time)}")

# Function to fetch pollution data in chunks
def fetch_pollution_data(start, end, chunk_days=5):
    all_pollution_data = []
    current_start = start
    
    while current_start < end:
        current_end = min(current_start + (chunk_days * 24 * 60 * 60), end)
        
        url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={current_start}&end={current_end}&appid={API_KEY}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            chunk_data = [
                {
                    "timestamp": item["dt"],
                    "aqi": item["main"]["aqi"],
                    "pm2_5": round(item["components"]["pm2_5"], 2),
                    "pm10": round(item["components"]["pm10"], 2),
                    "co": round(item["components"]["co"], 2)
                }
                for item in data.get("list", [])
            ]
            
            # Print the date range of the data we actually got back
            if chunk_data:
                first_date = datetime.fromtimestamp(chunk_data[0]["timestamp"])
                last_date = datetime.fromtimestamp(chunk_data[-1]["timestamp"])
                print(f"Got pollution data from {first_date} to {last_date}")
            
            all_pollution_data.extend(chunk_data)
        else:
            print(f"Failed to fetch pollution data for period {datetime.fromtimestamp(current_start)} to {datetime.fromtimestamp(current_end)}")
            print(f"Status code: {response.status_code}, Response: {response.text}")
        
        # Move to next chunk
        current_start = current_end
        # Add a delay to avoid hitting rate limits
        time.sleep(1)
    
    return all_pollution_data

# Function to fetch weather data in chunks
def fetch_weather_data(start, end, chunk_days=5):
    all_weather_data = []
    current_start = start
    
    while current_start < end:
        current_end = min(current_start + (chunk_days * 24 * 60 * 60), end)
        
        url = f"https://history.openweathermap.org/data/2.5/history/city?lat={LAT}&lon={LON}&type=hour&start={current_start}&end={current_end}&units=metric&appid={API_KEY}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            chunk_data = [
                {
                    "timestamp": hour["dt"],
                    "temp": round(hour["main"]["temp"], 2),
                    "humidity": round(hour["main"]["humidity"], 2)
                }
                for hour in data.get("list", [])
            ]
            
            # Print the date range of the data we actually got back
            if chunk_data:
                first_date = datetime.fromtimestamp(chunk_data[0]["timestamp"])
                last_date = datetime.fromtimestamp(chunk_data[-1]["timestamp"])
                print(f"Got weather data from {first_date} to {last_date}")
            
            all_weather_data.extend(chunk_data)
        else:
            print(f"Failed to fetch weather data for period {datetime.fromtimestamp(current_start)} to {datetime.fromtimestamp(current_end)}")
            print(f"Status code: {response.status_code}, Response: {response.text}")
        
        # Move to next chunk
        current_start = current_end
        # Add a delay to avoid hitting rate limits
        time.sleep(1)
    
    return all_weather_data

# Fetch data in chunks
print("Fetching pollution data...")
pollution_data = fetch_pollution_data(start_time, end_time)

print("Fetching weather data...")
weather_data = fetch_weather_data(start_time, end_time)

# Process pollution data
if pollution_data:
    df_pollution = pd.DataFrame(pollution_data)
    df_pollution['date'] = pd.to_datetime(df_pollution['timestamp'], unit='s').dt.date
    
    # Calculate means and round to 2 decimal places
    daily_pollution = df_pollution.groupby('date')[['aqi', 'pm2_5', 'pm10', 'co']].mean().round(2).reset_index()
    
    # Print date range in our processed data
    print(f"Pollution data date range: {daily_pollution['date'].min()} to {daily_pollution['date'].max()}")
else:
    print("No pollution data available.")
    daily_pollution = pd.DataFrame(columns=['date', 'aqi', 'pm2_5', 'pm10', 'co'])

# Process weather data - CHANGED to store min/max instead of mean
if weather_data:
    df_weather = pd.DataFrame(weather_data)
    df_weather['date'] = pd.to_datetime(df_weather['timestamp'], unit='s').dt.date
    
    # Get daily min and max values and round to 2 decimal places
    daily_temp_min = df_weather.groupby('date')['temp'].min().round(2).reset_index().rename(columns={'temp': 'temp_low'})
    daily_temp_max = df_weather.groupby('date')['temp'].max().round(2).reset_index().rename(columns={'temp': 'temp_high'})
    daily_humidity_min = df_weather.groupby('date')['humidity'].min().round(2).reset_index().rename(columns={'humidity': 'humidity_low'})
    daily_humidity_max = df_weather.groupby('date')['humidity'].max().round(2).reset_index().rename(columns={'humidity': 'humidity_high'})
    
    # Merge all weather data together
    daily_weather = pd.merge(daily_temp_min, daily_temp_max, on='date', how='inner', validate='one_to_one')
    daily_weather = pd.merge(daily_weather, daily_humidity_min, on='date', how='inner', validate='one_to_one')
    daily_weather = pd.merge(daily_weather, daily_humidity_max, on='date', how='inner', validate='one_to_one')
    
    # Print date range in our processed data
    print(f"Weather data date range: {daily_weather['date'].min()} to {daily_weather['date'].max()}")
else:
    print("No weather data available.")
    daily_weather = pd.DataFrame(columns=['date', 'temp_low', 'temp_high', 'humidity_low', 'humidity_high'])

# Merge data
if not daily_pollution.empty and not daily_weather.empty:
    merged_df = pd.merge(daily_pollution, daily_weather, on='date', how='inner', validate='one_to_one')
    
    # Check for odd dates in the final merged data
    print(f"Merged data date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    
    # Ensure all numeric values are rounded to 2 decimal places in final output
    numeric_columns = merged_df.select_dtypes(include=['float64']).columns
    merged_df[numeric_columns] = merged_df[numeric_columns].round(2)
    
    # Save the data
    merged_df.to_csv("daily_aqi_weather_90days.csv", index=False)
    print("Saved merged data to daily_aqi_weather_90days.csv")
else:
    print("Cannot merge dataframes because at least one is empty.")