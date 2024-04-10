# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /Preprocess

# Install system dependencies required for building certain Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ python3-dev libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN python3 -m spacy download en_core_web_md

# Copy the shell script and grant execution permissions
COPY run_preprocess.sh .
RUN chmod +x run_preprocess.sh

# Run the shell script when the container launches
CMD ["./run_preprocess.sh"]
