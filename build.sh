#!/bin/bash
# Force Python 3.11
export PYENV_VERSION=3.11.8
pip install --upgrade pip
pip install --only-binary=:all: -r requirements.txt
