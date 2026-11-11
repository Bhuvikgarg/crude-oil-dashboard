#!/usr/bin/env bash
# Start script for PetroSync Crude Oil Terminal
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

echo "=========================================================="
echo " Starting PetroSync Crude Oil Regime & Sentiment Terminal "
echo "=========================================================="

PORT=${PORT:-8080}
/home/bhuvik/anaconda3/bin/python app.py
