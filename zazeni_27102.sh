#!/bin/bash

# Nastavi ime slike
IMAGE_NAME="weequally/python-app"
TAG="latest"

# Prenos Docker slike
echo "Pulling Docker image..."
docker pull $IMAGE_NAME:$TAG

# Zagon Docker vsebnika
echo "Running Docker container..."
docker run -d -p 80:80 $IMAGE_NAME:$TAG
