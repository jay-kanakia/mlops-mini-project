#!/bin/bash
# Login to my AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 901619351636.dkr.ecr.us-east-1.amazonaws.com
docker push 901619351636.dkr.ecr.us-east-1.amazonaws.com/mini-project-ecr:v3
docker stop mini-project-app || true
docker rm mini-project-app || true
docker run -d -p 80:5000 -e DAGSHUB_PAT=e691c7193ab61dc9678e31c6b92ded8a65f80697 --name mini-project-app 901619351636.dkr.ecr.us-east-1.amazonaws.com/mini-project-ecr:v3