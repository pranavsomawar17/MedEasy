@echo off
setlocal

echo === Health Article Simplifier (Streamlit + Groq + FAISS) ===
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
echo Installing/verifying dependencies...
pip install -r requirements.txt

echo Starting Streamlit app...
streamlit run app.py
