#!/bin/bash
pip install --upgrade pip setuptools wheel
pip install --only-binary :all: -r requirements.txt
