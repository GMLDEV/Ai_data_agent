#!/bin/bash
# Setup script for Ollama with phi3 model

echo "Starting Ollama setup..."

# Wait for Ollama service to be ready
echo "Waiting for Ollama service..."
sleep 10

# Pull the phi3 model
echo "Pulling phi3 model..."
docker exec ollama ollama pull phi3

echo "Ollama setup complete!"
