# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /Dash

# Install system dependencies required for building certain Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ python3-dev libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy the Scrapy project to the working directory
COPY . .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint command to run the Scrapy spider
CMD python3 app.py