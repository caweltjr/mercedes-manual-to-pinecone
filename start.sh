#!/bin/bash
pip install openai==1.12.0 --force-reinstall --no-cache-dir
gunicorn app:app