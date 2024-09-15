FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

LABEL maintainer="Jongkuk Lim <limjk@jmarple.ai>"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV TERM xterm-256color

ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o user && useradd -m -u $UID -g $GID -o -s /bin/bash user

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y sudo dialog apt-utils tzdata
RUN echo "%sudo ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && echo "user:user" | chpasswd && adduser user sudo

WORKDIR /home/user
USER user

RUN sudo apt-get update && sudo apt-get install -y libgl1-mesa-dev && sudo apt-get -y install jq

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN sudo apt-get update && sudo apt-get -y install wget curl git
RUN curl -s https://raw.githubusercontent.com/JeiKeiLim/my_term/main/run.sh | /bin/bash

RUN sudo apt-get update && sudo apt-get install -y zsh && \
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
RUN echo "\n# Custom settings" >> /home/user/.zshrc && \
    echo "export PATH=/home/user/.local/bin:$PATH" >> /home/user/.zshrc && \
    echo "export LC_ALL=C.UTF-8 && export LANG=C.UTF-8" >> /home/user/.zshrc && \
    sed '11 c\ZSH_THEME=powerlevel10k/powerlevel10k' ~/.zshrc  > tmp.txt && mv tmp.txt ~/.zshrc && \
    echo 'POWERLEVEL9K_DISABLE_CONFIGURATION_WIZARD=true' >> ~/.zshrc
RUN sudo chsh -s $(which zsh) $(whoami)
# Place your environment here

RUN sudo apt-get install -y python3-pip && pip3 install --upgrade pip

RUN python3 -m pip install tensorboard wandb opencv-python
RUN sudo apt-get -y install mesa-common-dev libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev vulkan-utils mesa-vulkan-drivers libegl1 gcc-8 g++-8 pigz
RUN sudo rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json

RUN sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
RUN sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

COPY asset/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY asset/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY asset/IsaacGym_Preview_4_Package.tar.gz /home/user/
WORKDIR /home/user
RUN tar -xzf ~/IsaacGym_Preview_4_Package.tar.gz && rm ~/IsaacGym_Preview_4_Package.tar.gz
RUN cd isaacgym/python && \
    python3 -m pip install -e .

COPY --chown=user:user rsl_rl /home/user/rsl_rl
WORKDIR /home/user
RUN cd rsl_rl && \
    python3 -m pip install -e .

COPY --chown=user:user legged_gym /home/user/legged_gym
WORKDIR /home/user
RUN cd legged_gym && \
    python3 -m pip install -e .

RUN sudo apt-get -y install python3-actionlib
COPY requirements.txt /home/user/
WORKDIR /home/user
# RUN python3 -m pip install -r requirements.txt && rm requirements.txt

