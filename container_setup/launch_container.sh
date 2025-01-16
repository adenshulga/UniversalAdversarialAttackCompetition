#! /bin/bash
source credentials

docker run \
    -d \
    --shm-size=8g \
    --memory=8g \
    --cpuset-cpus=96-107 \
    --user ${DOCKER_USER_ID}:${DOCKER_GROUP_ID} \
    --name ${CONTAINER_NAME} \
    --rm \
    -it \
    --init \
    --gpus '"device=3"' \
    -v /home/${USER}/${SRC}:/app \
    -p ${INNER_PORT}:${CONTAINER_PORT} \
    ${DOCKER_NAME} \
    bash
