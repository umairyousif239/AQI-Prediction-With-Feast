import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os
import datetime

def evaluate_model(model, x_test, y_test, model_name):
    """Evaluate model performance and return metrics"""
    y_pred = model.predict(x_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return {
        'model_name': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }

def plot_actual_vs_predicted(y_test, predictions_dict, save_path=None):
    """Plot actual vs. predicted values for all models"""
    plt.figure(figsize=(12, 8))
    
    # Plot actual values
    plt.plot(range(len(y_test)), y_test.values, 'b-', label='Actual Values', linewidth=2)
    
    # Plot predicted values for each model
    colors = ['r-', 'g-', 'm-']
    for (model_name, preds), color in zip(predictions_dict.items(), colors):
        plt.plot(range(len(preds)), preds, color, label=f'{model_name} Predictions', linewidth=1, alpha=0.7)
    
    plt.title('Actual vs Predicted AQI Values', fontsize=16)
    plt.xlabel('Sample Index', fontsize=14)
    plt.ylabel('AQI Value', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_feature_importance(model, feature_names, model_name, save_path=None):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

def main():
    print("Loading AQI training features...")
    
    # Load the dataset
    df = pd.read_csv("feature_repo/aqi_training_features.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    
    # Convert event_timestamp to datetime and check for missing values
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Data exploration
    print(f"\nData description:\n{df.describe()}")
    
    # Drop timestamp
    X = df.drop(['event_timestamp', 'aqi'], axis=1)  # Features
    y = df['aqi']  # Target variable
    
    # Create a directory for saving results
    results_dir = "feature_repo/model_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Split the data (80% train, 20% test)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining set shape: {x_train.shape}")
    print(f"Testing set shape: {x_test.shape}")
    
    # Feature scaling
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    # Save the scaler for future use
    joblib.dump(scaler, f"{results_dir}/scaler.pkl")
    
    # List of models to test
    models = [
        ('Linear Regression', LinearRegression()),
        ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ]
    
    # Train, evaluate and store results
    results = []
    predictions_dict = {}
    
    for name, model in models:
        print(f"\nTraining {name}...")
        model.fit(x_train_scaled, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, x_train_scaled, y_train, cv=5, scoring='r2')
        print(f"Cross-validation R² scores: {cv_scores}")
        print(f"Mean CV R² score: {cv_scores.mean():.4f}")
        
        # Evaluate on test set
        eval_results = evaluate_model(model, x_test_scaled, y_test, name)
        results.append(eval_results)
        predictions_dict[name] = eval_results['predictions']
        
        # Save the model
        joblib.dump(model, f"{results_dir}/{name.lower().replace(' ', '_')}_model.pkl")
        
        # Plot feature importance for tree-based models
        if name in ['Random Forest', 'Gradient Boosting']:
            plot_feature_importance(
                model, 
                X.columns.tolist(), 
                name,
                f"{results_dir}/{name.lower().replace(' ', '_')}_feature_importance.png"
            )
    
    # Plot actual vs predicted values
    plot_actual_vs_predicted(
        y_test, 
        predictions_dict,
        f"{results_dir}/actual_vs_predicted.png"
    )
    
    # Compare models
    print("\nModel Comparison:")
    comparison_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'RMSE': r['rmse'],
            'MAE': r['mae'],
            'R²': r['r2']
        } for r in results
    ])
    print(comparison_df)
    
    # Save comparison results
    comparison_df.to_csv(f"{results_dir}/model_comparison.csv", index=False)
    
    # Find the best model based on R²
    best_model_idx = comparison_df['R²'].idxmax()
    best_model_name = comparison_df.loc[best_model_idx, 'Model']
    print(f"\nBest performing model: {best_model_name}")
    
    print(f"\nAll model results and visualizations saved to {results_dir}/")

if __name__ == "__main__":
    main()