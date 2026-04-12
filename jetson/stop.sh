#!/usr/bin/env bash
# stop.sh — Kill all running microservices
pkill -f "uvicorn jetson.services" 2>/dev/null
pkill -f "http.server 5173"        2>/dev/null
echo "Services stopped."
