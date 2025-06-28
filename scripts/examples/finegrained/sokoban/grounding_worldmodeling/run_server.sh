#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Interactive input for port and CUDA devices
read -p "Enter port number (default: 5000): " PORT_INPUT
PORT=${PORT_INPUT:-5000}

read -p "Enter CUDA devices (default: 0,1,2,3): " CUDA_DEVICES
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3}

# Start the server
python -m vagen.server.server server.port=$PORT use_state_reward=False