import os
import argparse
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from joblib import Parallel, delayed, load
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from feast import FeatureStore
import shap

# Configuration
CONFIG = {
    'TIMEZONE': timezone.utc,
    'DATA_PATH': 'aqi_training_features.csv',
    'RESULTS_DIR': 'model_results',
    'PREDICTIONS_DIR': 'predictions',
    'COMPARISON_FILE': 'model_comparison.csv',
    'SCALER_FILE': 'scaler.pkl',
    'FORECAST_DAYS': 3,
    'ONLINE_FEATURES': [
        'air_quality_metrics:pm2_5',
        'air_quality_metrics:pm10',
        'air_quality_metrics:co',
        'weather_metrics:temp_low',
        'weather_metrics:temp_high',
        'weather_metrics:humidity_low',
        'weather_metrics:humidity_high',
    ],
}
FEATURE_COLS = ['pm2_5', 'pm10', 'co', 'temp_low', 'temp_high', 'humidity_low', 'humidity_high']
# Units mapping for feature plots
UNITS = {
    'pm2_5': 'μg/m³',
    'pm10': 'μg/m³',
    'co': 'ppm',
    'temp_low': '°C',
    'temp_high': '°C',
    'humidity_low': '%',
    'humidity_high': '%',
}

# Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Utility Functions
def localize_timestamps(df: pd.DataFrame, tz: timezone) -> pd.DataFrame:
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
    if df['event_timestamp'].dt.tz is None:
        return df.assign(event_timestamp=df['event_timestamp'].dt.tz_localize(tz))
    return df.assign(event_timestamp=df['event_timestamp'].dt.tz_convert(tz))


def load_data(path: str, tz: timezone) -> pd.DataFrame:
    df = pd.read_csv(path)
    return localize_timestamps(df, tz)


def load_best_model(results_dir: str) -> Pipeline:
    comp_path = os.path.join(results_dir, CONFIG['COMPARISON_FILE'])
    if not os.path.exists(comp_path):
        raise FileNotFoundError(f"Comparison file not found: {comp_path}")
    comp_df = pd.read_csv(comp_path)
    best = comp_df.loc[comp_df['R²'].idxmax(), 'Model']
    model_file = f"{best.lower().replace(' ', '_')}_model.pkl"
    model_path = os.path.join(results_dir, model_file)
    scaler_path = os.path.join(results_dir, CONFIG['SCALER_FILE'])
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler file missing.")
    model = load(model_path)
    scaler = load(scaler_path)
    pipeline = Pipeline([('scaler', scaler), ('regressor', model)])
    logger.info(f"Loaded model and scaler from {results_dir}")
    return pipeline, best

# Feature Retrieval
def get_historical_features(df: pd.DataFrame, days: int) -> pd.DataFrame:
    df = df.sort_values('event_timestamp')
    end = df['event_timestamp'].max()
    start = end - timedelta(days=days)
    recent = df[df['event_timestamp'] >= start].copy()
    logger.info(f"Historical data from {start.date()} to {end.date()}")
    return recent


def get_latest_features(store: FeatureStore,
                        historical_df: pd.DataFrame,
                        num_days: int) -> Optional[pd.DataFrame]:
    today = datetime.now(CONFIG['TIMEZONE']).date()
    last_hist = historical_df['event_timestamp'].max().date()
    start_date = max(last_hist + timedelta(days=1), today)
    dates = [start_date - timedelta(days=i) for i in range(num_days)]
    valid_dates = [d for d in dates if d <= today]
    if not valid_dates:
        logger.warning("No valid dates for online features.")
        return None
    entity_rows = [{'date_key': d.strftime('%Y-%m-%d')} for d in valid_dates]
    logger.info(f"Retrieving online features for dates: {[r['date_key'] for r in entity_rows]}")
    of = store.get_online_features(
        features=CONFIG['ONLINE_FEATURES'],
        entity_rows=entity_rows
    ).to_dict()
    df = pd.DataFrame(of)
    df['event_timestamp'] = pd.to_datetime(df['date_key']).dt.tz_localize(CONFIG['TIMEZONE'])
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    if len(df) < before:
        missing = set(r['date_key'] for r in entity_rows) - set(df['date_key'])
        logger.warning(f"Dropped dates with missing features: {missing}")
    return df

# Forecasting
def train_feature_models(df: pd.DataFrame) -> Dict[str, Dict]:
    df = df.copy()
    df['dayofweek'] = df['event_timestamp'].dt.dayofweek
    df['month'] = df['event_timestamp'].dt.month
    df['day'] = df['event_timestamp'].dt.day
    for col in FEATURE_COLS:
        for lag in (1, 2, 3):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    df = df.dropna()

    def fit_one(col: str):
        features = ['dayofweek', 'month', 'day'] + [f'{col}_lag{i}' for i in (1, 2, 3)]
        model = GradientBoostingRegressor(n_estimators=100,
                                          learning_rate=0.1,
                                          max_depth=3,
                                          random_state=42)
        model.fit(df[features], df[col])
        logger.info(f"Trained model for {col}")
        return col, {'model': model, 'features': features}

    results = Parallel(n_jobs=-1)(delayed(fit_one)(c) for c in FEATURE_COLS)
    return dict(results)


def forecast_features(latest_df: pd.DataFrame,
                      models: Dict[str, Dict],
                      days: int) -> pd.DataFrame:
    base = latest_df.sort_values('event_timestamp').copy()
    last_date = base['event_timestamp'].max()
    rows = []
    for i in range(1, days + 1):
        date = last_date + timedelta(days=i)
        row = {
            'date_key': date.strftime('%Y-%m-%d'),
            'event_timestamp': date,
            'dayofweek': date.dayofweek,
            'month': date.month,
            'day': date.day
        }
        for col in FEATURE_COLS:
            feats = models[col]['features']
            lags = {f'{col}_lag{lag}': base[col].iloc[-lag] for lag in (1, 2, 3)}
            inp = pd.DataFrame([{**{f: row[f] for f in feats[:3]}, **lags}])
            val = models[col]['model'].predict(inp)[0]
            row[col] = round(max(0, val), 2)
        rows.append(row)
        base = pd.concat([base, pd.DataFrame([row])], ignore_index=True)
    return pd.DataFrame(rows)

# Prediction & Saving
def predict_aqi(pipeline: Pipeline,
                df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['predicted_aqi'] = pipeline.predict(df[FEATURE_COLS]).round(2)
    return df

def save_predictions(df: pd.DataFrame, filepath: str) -> None:
    cols = ['date_key', 'event_timestamp', 'predicted_aqi'] + FEATURE_COLS
    out_df = df[cols].sort_values('date_key')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    out_df.to_csv(filepath, index=False)
    logger.info(f"Saved cleaned predictions to {filepath}")

# Visualization
def plot_aqi(df: pd.DataFrame, model_name: str, outpath: str) -> None:
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    today = datetime.now(CONFIG['TIMEZONE']).replace(hour=0, minute=0, second=0, microsecond=0)
    actual = df[df['event_timestamp'] <= today]
    forecast = df[df['event_timestamp'] > today]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual['event_timestamp'], actual['predicted_aqi'], marker='o', label='Predicted AQI (actual)')
    if not forecast.empty:
        ax.plot(forecast['event_timestamp'], forecast['predicted_aqi'], marker='o', linestyle='--', label='Forecast AQI')
    for level in range(1, 6):
        ax.axhline(level, linestyle='--', alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.set(title=f'AQI Predictions ({model_name})', xlabel='Date', ylabel='AQI')
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    logger.info(f"AQI forecast saved to {outpath}")


def plot_features(df: pd.DataFrame, out_dir: str) -> None:
    today = datetime.now(CONFIG['TIMEZONE']).replace(hour=0, minute=0, second=0, microsecond=0)
    for feature in FEATURE_COLS:
        actual = df[df['event_timestamp'] <= today]
        forecast = df[df['event_timestamp'] > today]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(actual['event_timestamp'], actual[feature], marker='o', label=f'{feature} (actual)')
        if not forecast.empty:
            ax.plot(forecast['event_timestamp'], forecast[feature], marker='o', linestyle='--', label=f'{feature} (forecast)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.set(title=f'{feature.upper()} Forecast', xlabel='Date', ylabel=f'{feature} ({UNITS.get(feature, "")})')
        ax.legend()
        fig.tight_layout()
        filepath = os.path.join(out_dir, f"{feature}_forecast.png")
        fig.savefig(filepath)
        plt.close(fig)
        logger.info(f"{feature} forecast plot saved to {filepath}")

# SHAP Integration
def explain_model_with_shap(pipeline: Pipeline, df: pd.DataFrame, out_dir: str) -> None:
    """
    Use SHAP to explain the dependency through visualizations.

    Args:
        pipeline (Pipeline): The trained model pipeline.
        df (pd.DataFrame): The dataset used for predictions.
        out_dir (str): Directory to save SHAP visualizations.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Extract the model and scaler from the pipeline
    scaler = pipeline.named_steps['scaler']
    model = pipeline.named_steps['regressor']

    # Scale the feature data
    X = scaler.transform(df[FEATURE_COLS])

    # Initialize SHAP explainer
    explainer = shap.Explainer(model, X)

    # Calculate SHAP values
    shap_values = explainer(X)

    # Save SHAP summary plot
    summary_plot_path = os.path.join(out_dir, 'shap_summary_plot.png')
    shap.summary_plot(shap_values, X, feature_names=FEATURE_COLS, show=False)
    plt.savefig(summary_plot_path)
    plt.close()
    logger.info(f"SHAP summary plot saved to {summary_plot_path}")

    # Save SHAP dependence plots for each feature
    for feature in FEATURE_COLS:
        dependence_plot_path = os.path.join(out_dir, f'shap_dependence_{feature}.png')
        shap.dependence_plot(feature, shap_values.values, X, feature_names=FEATURE_COLS, show=False)
        plt.savefig(dependence_plot_path)
        plt.close()
        logger.info(f"SHAP dependence plot for {feature} saved to {dependence_plot_path}")

# Main Entry
def main():
    parser = argparse.ArgumentParser(description="AQI Prediction Pipeline")
    parser.add_argument('--forecast-days', type=int,
                        default=CONFIG['FORECAST_DAYS'],
                        help='Days to forecast features')
    args = parser.parse_args()

    df = load_data(CONFIG['DATA_PATH'], CONFIG['TIMEZONE'])
    hist = get_historical_features(df, days=85)

    store = FeatureStore(repo_path='')
    latest = get_latest_features(store, hist, num_days=30)

    feat_models = train_feature_models(hist)
    forecast_df = forecast_features(latest, feat_models, days=args.forecast_days)
    all_feats = pd.concat([latest, forecast_df], ignore_index=True)

    pipeline, model_name = load_best_model(CONFIG['RESULTS_DIR'])
    preds = predict_aqi(pipeline, all_feats)

    out_dir = CONFIG['PREDICTIONS_DIR']
    out_csv = os.path.join(out_dir, 'aqi_predictions.csv')
    save_predictions(preds, out_csv)
    plot_aqi(preds, model_name, os.path.join(out_dir, 'aqi_forecast.png'))
    plot_features(all_feats, out_dir)

    shap_dir = os.path.join(out_dir, 'shap_explanation')
    explain_model_with_shap(pipeline, all_feats, shap_dir)

if __name__ == '__main__':
    main()
