#!/bin/bash

if [ "$1" -eq 1 ]; then
    cd main
else
    cd api
fi

# Define the Python script you want to run
PYTHON_SCRIPT="api.py"
VENV_NAME=".venv"

# Define the log file
LOG_FILE="nohup.out"

# Kill the existing Python process
pkill -f "$PYTHON_SCRIPT"

if [ ! -d VENV_NAME ]; then 
    python3 -m venv .venv 
fi

# Activate virtual environment
source .venv/bin/activate

pip install -r requirements.txt

# Run the Python script in the background with nohup
nohup python "$PYTHON_SCRIPT" > "$LOG_FILE" 2>&1 &

# Get the PID of the last background process
PID=$!

# Print the PID
echo "Python script started with PID: $PID"
