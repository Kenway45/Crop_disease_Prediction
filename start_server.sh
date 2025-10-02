#!/bin/bash

# Start Demo Server Script
cd "$(dirname "$0")"

echo "🚀 Starting Crop Disease Detection Server..."
echo ""

# Kill any existing server
pkill -9 -f demo_server.py 2>/dev/null
sleep 1

# Get IP address
IP=$(ifconfig | grep "inet " | grep -v "127.0.0.1" | awk '{print $2}' | grep -E "^(192|172|10)\." | head -1)

if [ -z "$IP" ]; then
    IP="localhost"
    echo "⚠️  Warning: Could not detect network IP"
    echo "You may need to check your WiFi connection"
    echo ""
fi

echo "============================================================"
echo "  CROP DISEASE DETECTION SERVER"
echo "============================================================"
echo ""
echo "📱 ON YOUR PHONE, open this URL:"
echo ""
echo "     http://$IP:8080"
echo ""
echo "============================================================"
echo ""
echo "💻 On this computer, use:"
echo "     http://localhost:8080"
echo ""
echo "🔥 Server starting..."
echo ""

# Start the server
python3 src/demo_server.py

echo ""
echo "Server stopped."

