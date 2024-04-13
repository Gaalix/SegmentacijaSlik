#!/bin/bash

# Nastavi ime slike
IMAGE_NAME="weequally/python-app"
TAG="latest"

# Prijava v DockerHub z uporabo tokena
echo "Logging into DockerHub..."
echo "$DOCKER_TOKEN" | docker login --username $DOCKER_USERNAME --password-stdin

# Gradnja Docker slike
echo "Building Docker image..."
docker build -t $IMAGE_NAME:$TAG .

# Potiskanje slike na DockerHub
echo "Pushing image to DockerHub..."
docker push $IMAGE_NAME:$TAG
