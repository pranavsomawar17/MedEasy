#!/usr/bin/env bash
set -e

echo "=== Health Article Simplifier (Streamlit + Groq + FAISS) ==="
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate
echo "Installing/verifying dependencies..."
pip install -r requirements.txt

echo "Starting Streamlit app..."
streamlit run app.py
