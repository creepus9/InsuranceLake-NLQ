#!/bin/sh

# Builds three Docker images for Natural Language Query (NLQ) demo using Amazon Athena
# Amazon Bedrock
# run: chmod a+rx build.sh
# sh ./build.sh

# Value located in the output from the nlq-genai-infra CloudFormation template
# e.g. 111222333444.dkr.ecr.us-east-1.amazonaws.com/nlq-genai
ECS_REPOSITORY="<you_ecr_repository>"

aws ecr get-login-password --region us-east-1 |
	docker login --username AWS --password-stdin $ECS_REPOSITORY


# Deploy with Amazon Bedrock
TAG="2.0.0-bedrock"
docker build -f Dockerfile_Bedrock -t $ECS_REPOSITORY:$TAG .
docker push $ECS_REPOSITORY:$TAG

docker image ls
