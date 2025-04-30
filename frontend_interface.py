import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import threading
import uvicorn
import os
import sys
import subprocess
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time

# ===================== FASTAPI BACKEND =====================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/aqi-data")
def get_aqi_data():
    try:
        file_path = 'feature_repo/predictions/aqi_predictions.csv'
        if not os.path.exists(file_path):
            file_path = '../feature_repo/predictions/aqi_predictions.csv'
        df = pd.read_csv(file_path)
        return df.tail(4).to_dict(orient="records")
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000)

# ===================== STREAMLIT APP CODE =====================
def create_streamlit_app():
    streamlit_code = """  # STREAMLIT CODE OMITTED FOR BREVITY (unchanged) """
    with open("streamlit_app.py", "w", encoding="utf-8") as f:
        f.write(streamlit_code)
    return "streamlit_app.py"

# ===================== MAIN EXECUTION =====================
if __name__ == "__main__":
    def kill_after_timeout(proc, timeout=300):
        time.sleep(timeout)
        print(f"⏳ Timeout reached ({timeout}s). Terminating Streamlit...")
        proc.terminate()

    # Start FastAPI
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    print("Starting FastAPI server on http://127.0.0.1:8000")

    # Create and start Streamlit
    streamlit_file = create_streamlit_app()
    time.sleep(1)
    print("Starting Streamlit frontend...")

    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", streamlit_file])

    # Start thread to kill Streamlit after 5 minutes
    killer_thread = threading.Thread(target=kill_after_timeout, args=(proc,))
    killer_thread.start()

    # Wait for Streamlit to finish
    proc.wait()
    print("✅ Streamlit process exited.")
