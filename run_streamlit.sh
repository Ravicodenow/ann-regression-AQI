#!/bin/bash

echo "🌬️ Starting Air Quality Index Prediction App"
echo "==============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r streamlit_requirements.txt

# Start Streamlit app
echo "🚀 Starting Streamlit app..."
echo "🌐 App will be available at http://localhost:8501"
echo "💡 Press Ctrl+C to stop the app"

streamlit run streamlit_app.py