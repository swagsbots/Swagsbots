#!/bin/bash
# Find Python executable
PYTHON_CMD=$(which python3 || which python)

if [ -z "$PYTHON_CMD" ]; then
    echo "Python not found! Available options:"
    ls /usr/bin/python*
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
echo "Python version:"
$PYTHON_CMD --version

# Run the application
$PYTHON_CMD bitget_trading_bot.py
