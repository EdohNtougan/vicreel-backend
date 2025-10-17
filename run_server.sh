#!/bin/bash
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8080 --log-level info
