FROM python:3.12-slim

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

ARG DEBIAN_FRONTEND=noninteractive
# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    sudo \
    git \
    unzip \
    tmux \
    vim \
    fish \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

ARG DOCKER_NAME
ARG DOCKER_USER_ID
ARG DOCKER_GROUP_ID

USER root
RUN groupadd -g $DOCKER_GROUP_ID $DOCKER_NAME

RUN adduser --disabled-password --uid $DOCKER_USER_ID --gid $DOCKER_GROUP_ID --gecos '' --shell /bin/bash $DOCKER_NAME \
    && chown -R $DOCKER_NAME:$DOCKER_NAME /app
RUN echo "$DOCKER_NAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-$DOCKER_NAME
USER $DOCKER_NAME

ENV HOME=/home/$DOCKER_NAME

RUN mkdir -p $HOME/.cache $HOME/.config $HOME/tmp $HOME/opt/conda\
    && chmod -R 777 $HOME

EXPOSE 8000

CMD ["bash"]
