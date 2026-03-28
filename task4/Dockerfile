# Use a base image with Python 3.9
FROM python:3.9-slim-buster

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Update sources.list to use Debian archive, then update package lists and install system dependencies
RUN sed -i 's|http://deb.debian.org/debian|http://archive.debian.org/debian|g' /etc/apt/sources.list \
    && sed -i 's|http://security.debian.org/debian-security|http://archive.debian.org/debian-security|g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Install Python dependencies
# TensorFlow Quantum 0.7.6 is compatible with TensorFlow 2.8-2.10
# Python 3.9 is compatible with TensorFlow 2.10
RUN pip install --no-cache-dir \
    tensorflow==2.8.0 \
    tensorflow-quantum==0.7.2 \
    cirq==0.13.1 \
    numpy==1.23.5 \
    pandas==1.5.3 \
    matplotlib==3.7.0 \
    scikit-learn==1.2.2

# Copy your project files into the container
# If you have your QGAN script (e.g., qgan_model.py), copy it here
# Copy all files from task4 into the container
COPY . ./

# Command to run your application (runs task4/qgan.py)
CMD ["python", "task4/qgan.py"]
