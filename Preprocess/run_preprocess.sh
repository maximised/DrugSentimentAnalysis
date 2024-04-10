#!/bin/bash
# Stop on error
set -e

# Run preprocessing scripts in order
python3 "1_preprocess_webmd.py"
python3 "2_sentiment.py"
python3 "3_topic_model.py"
