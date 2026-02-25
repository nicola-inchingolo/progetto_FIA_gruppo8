# Use an official Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to install dependencies
COPY requirements.txt .

# Install system dependencies required by some graphical libraries and Python dependencies
RUN apt-get update && apt-get install -y \
    && pip install --no-cache-dir -r requirements.txt


# Copy the rest of the source code into the container
COPY . .

# Create input and output directories (they will be used as mount points)
RUN mkdir -p /app/data /app/output_plots /app/output_result

# Command to start the application.
# The -u flag forces unbuffered output (useful to see logs in real time)
CMD ["python", "-u", "main.py"]
