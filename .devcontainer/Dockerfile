# Base image: Use your specified NVIDIA CUDA image
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

# Install necessary tools and add the 'vscode' user
RUN apt-get update && apt-get install -y sudo git openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Create the vscode user with a home directory
RUN useradd -ms /bin/bash vscode && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers