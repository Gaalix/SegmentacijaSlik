# Use an official Python runtime as a parent image
FROM ubuntu:latest

RUN apt-get update -y && apt-get install -y python3 python3-pip

RUN pip install --upgrade pip

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run app.py when the container launches
CMD ["python3", "naloga3.py"]
