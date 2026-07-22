#!/bin/bash

# Kill any existing server running on port 8080
if lsof -i :8080 >/dev/null 2>&1; then
    echo "Stopping existing server on port 8080..."
    kill -9 $(lsof -t -i:8080)
fi

echo "🚀 Starting EOD Streamlit Dashboard..."
python3 -m streamlit run src/trading/dashboard_streamlit.py --server.port 8080 --server.headless true &

echo "⚡ Dashboard is loading! Visit: http://localhost:8080"
