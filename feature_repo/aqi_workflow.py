import subprocess
from datetime import datetime, timedelta, timezone
import pandas as pd
from feast import FeatureStore

def run_aqi_workflow():
    # Initialize the feature store
    store = FeatureStore(repo_path="")
    
    print("\n--- Run feast apply ---")
    subprocess.run(["feast", "apply"])

    print("\n--- Load features into online store ---")
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=90)
    store.materialize(start_date=start_date, end_date=end_date)

    print("\n--- Checking for data gaps ---")
    df = pd.read_parquet("data/daily_aqi_weather_90days.parquet")
    print(f"Available date range: {df['date_key'].min()} to {df['date_key'].max()}")
    print(f"Total dates in data: {len(df['date_key'].unique())}")

    print("\n--- Retrieve historical features for model training ---")
    df = pd.read_parquet("data/daily_aqi_weather_90days.parquet")
    entity_df = pd.DataFrame({
        "date_key": df["date_key"].values,
        "event_timestamp": df["timestamp_key"].values
    })

    # Fetch historical features
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "air_quality_metrics:aqi",
            "air_quality_metrics:pm2_5",
            "air_quality_metrics:pm10",
            "air_quality_metrics:co",
            "weather_metrics:temp_low",
            "weather_metrics:temp_high",
            "weather_metrics:humidity_low",
            "weather_metrics:humidity_high",
        ],
    ).to_df()

    # Drop duplicate or conflicting columns
    if "date_key" in training_df.columns:
        training_df = training_df.drop(columns=["date_key"])

    print("Historical features retrieved:")
    print(training_df.head())
    
    # Save features for model training
    training_df.to_csv("aqi_training_features.csv", index=False)
    print("Training features saved to aqi_training_features.csv")

    print("\n--- Retrieve online features for a specific date ---")
    # Retrieve online features for a specific date
    entity_rows = [{"date_key": datetime.now(timezone.utc).strftime('%Y-%m-%d')}]
    
    returned_features = store.get_online_features(
        features=[
            "air_quality_metrics:aqi",
            "air_quality_metrics:pm2_5",
            "air_quality_metrics:pm10",
            "air_quality_metrics:co",
            "weather_metrics:temp_low",
            "weather_metrics:temp_high",
            "weather_metrics:humidity_low",
            "weather_metrics:humidity_high",
        ],
        entity_rows=entity_rows,
    ).to_dict()
    
    print("Online features retrieved:")
    for key, value in sorted(returned_features.items()):
        print(key, " : ", value)

if __name__ == "__main__":
    run_aqi_workflow()