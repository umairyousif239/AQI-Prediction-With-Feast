from datetime import timedelta
import pandas as pd
from feast import (
    Entity, 
    FeatureView, 
    Field, 
    FileSource,
    ValueType
)
from feast.types import Float32, Int32

# Define the date entity
date_entity = Entity(
    name="date_key",
    join_keys=["date_key"],
    value_type=ValueType.STRING,
    description="Calendar date for AQI and weather measurements",
)

# Define the data source for your parquet file with event_timestamp_column
aqi_weather_source = FileSource(
    path="data/daily_aqi_weather_90days.parquet",
    event_timestamp_column="timestamp_key"
)

# Define feature view for air quality metrics
air_quality_features = FeatureView(
    name="air_quality_metrics",
    entities=[date_entity],
    ttl=timedelta(days=90),  # Time-to-live for feature values
    schema=[
        Field(name="aqi", dtype=Int32),
        Field(name="pm2_5", dtype=Float32),
        Field(name="pm10", dtype=Float32),
        Field(name="co", dtype=Float32),
    ],
    source=aqi_weather_source,
    online=True,
    description="Daily air quality metrics including AQI, PM2.5, PM10, and CO levels",
)

# Define feature view for weather metrics
weather_features = FeatureView(
    name="weather_metrics",
    entities=[date_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="temp_low", dtype=Float32),
        Field(name="temp_high", dtype=Float32),
        Field(name="humidity_low", dtype=Float32),
        Field(name="humidity_high", dtype=Float32),
    ],
    source=aqi_weather_source,
    online=True,
    description="Daily weather metrics including temperature and humidity ranges",
)