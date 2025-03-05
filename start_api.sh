#!/bin/bash

# Define the Python script you want to run
PYTHON_SCRIPT="api.py"

# Define the log file
LOG_FILE="nohup.out"

# Kill the existing Python process
pkill -f "$PYTHON_SCRIPT"

# Run the Python script in the background with nohup
nohup .venv/bin/python "$PYTHON_SCRIPT" > "$LOG_FILE" 2>&1 &

# Get the PID of the last background process
PID=$!

# Print the PID
echo "Python script started with PID: $PID"
