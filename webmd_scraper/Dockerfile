# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /webmd_scraper

# Copy the Scrapy project to the working directory
COPY . .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint command to run the Scrapy spider
CMD scrapy crawl webmd