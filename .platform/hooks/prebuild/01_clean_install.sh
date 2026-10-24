#!/bin/bash
set -e

source /var/app/venv/staging-LQM1lest/bin/activate
pip cache purge 2>/dev/null || true
pip install --no-cache-dir -r /var/app/staging/requirements.txt
