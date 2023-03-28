export AWS_ACCOUNT_ID=371380984152
export URL_STRING=".dkr.ecr.ap-southeast-1.amazonaws.com"

CONTAINER_STRING="smu-iot"
IMAGE_STRING="latest"
ECR_IMAGE_URI="$AWS_ACCOUNT_ID$URL_STRING/$CONTAINER_STRING:$IMAGE_STRING"

aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID$URL_STRING"
# remove previous images to save space
docker rmi "$AWS_ACCOUNT_ID$URL_STRING/$CONTAINER_STRING"
docker rmi "$CONTAINER_STRING"
# build image
docker build --tag "$CONTAINER_STRING" .
# tag and push to AWS ECR
docker tag $CONTAINER_STRING:latest "$ECR_IMAGE_URI"
docker push "$ECR_IMAGE_URI"

