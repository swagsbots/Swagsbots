#!/bin/bash
# Find Python explicitly
PYTHON_CMD=$(which python3.11 || which python3 || which python)
echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD bitget_trading_bot.py
