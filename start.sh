#!/bin/sh
set -e  # Stop on any error

echo "Running api_data_fetch.py..."
python api_data_fetch.py

cd feature_repo

echo "Running feast apply..."
feast apply

echo "Running aqi workflow..."
python "aqi_workflow.py"

cd ..

echo "comparing models..."
python model_training.py

cd feature_repo

echo "Training model..."
python aqi_prediction.py

cd ..

echo "launching the frontend interface..."
python frontend_interface.py