# Accelerators

Welcome to GPU Accelerators course.

# Setting up environments

Below is an incomplete list of possible environment setups. If your preferred setup is not found you have to look online for instructions how to set it up.

## Prerequisites

It's assumed that:
- your OS system is setup and up to date
- you have latest GPU drivers

## Conda or Virtual Environments?

Conda is a package manager and will help you with managing package dependencies. It is useful for installing eg. python or cuda runtimes - so something that python virtual environments (venvs) cannot do.

It can be problematic to use both conda and virtual environments. So depending on your OS platform, accelerations platform or your preferences, it may be more convenient to use conda or venvs.

- When you installed packages and libraries with Conda - stick to conda envs
- When you installed packages like CUDA and python manually use python venvs

### WSL2 Ubuntu setup

DO NOT:
- don't install any drivers in WSL2 - Nvidia Windows drivers handle all work
- don't install CUDA system wide with sudo apt - there are many confusing dependencies, so installing it manually is problematic

You need to install WSL

```wsl.exe --install```
```wsl.exe --update```

Create Ubuntu 24.04 LTS - you can do it in VS Code or get it from Microsoft Store.

Once WSL2 Ubuntu is created update it and install build packages.

```sudo apt update```
```sudo apt upgrade```
```sudo apt install -y build-essential```

### Install Conda

Download installation script

```wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh```

Then run and answer the questions when prompted - you can install it in user directory if working alone on a machine

```bash Miniconda3-latest-Linux-x86_64.sh```

Create new conda env

```conda create -n <env_name> python=<version>```

As of 2025H2 choose python 3.11 for maximum compatibility.

```conda create -n gpu python=3.11```

```conda activate gpu```

## OpenCL

### PyOpenCL on Windows

When install python bindings for OpenCL

```pip install pyopencl```

### PyOpenCL on Ubuntu with WSL2

Seems like this is not supported, due to Nvidia driver not supporting OpenCL in WSL2.

## CUDA

### PyCuda on Windows

Install MSVC - Microsoft compiler - needed to compile CUDA kernels
- download Visual Studio eg. Community version - latest is 2022
- during installation make sure  "MSVC - VS 2022 C++ build tools" is selected

Assuming you use virtual environment, add the snippet from ```CUDA\activate_bat``` to your ```.venv\Scripts\activate.bat``` before the ```:END``` clause. This is required so that VS development prompt is setup and ```cl.exe``` compiler and other paths are setup correctly. Alternatively VS Code can be run from a propmpt initialized with the content below but it leave a command window open:
```
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
code c:\projects\accelerators
```

Install CUDA Toolkit from Nvidia's website installer.

Intall CUDA from Nvidia's website installer. CUDA toolkit version 13.0 does not work with PyCuda 2025.1.2.
Tested combination:
- CUDA 12.8
- PyCuda 2025.1.2

Add location of CUDA DLLs to your path. Executables should be added during installation. Check the following are in the System or User Paths.

```C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin```
```C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp```
```C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\x64```

Then install PyCuda

```pip install pycuda```

### CUDA on Ubuntu - system wide install

Install relevant CUDA by following instructions on [NVidia's CUDA website](https://developer.nvidia.com/cuda-downloads). The installation on Ubuntu is to download a Debian package which installs the CUDA repository signing key on your system, allowing you to securely add and update NVIDIA CUDA packages from their official repository using your package manager (like apt). Then you will use ```apt-get``` to download requested cuda toolkit. 

```sudo apt-get -y install cuda-toolkit-12-8```

It will contain working ```cuda-gdb``` for native code debugging.

### Native CUDA C/C++ compilation and debugging setup

#### Install CMake

Install cmake with 

```sudo apt install cmake```

#### VS Code extenstions

If you use VS Code for development install the following extensions:
- C/C++ - ```ms-vscode.cpptools```
- Nsight Visual Studio Code Edition - ```nvidia.nsight-vscode-edition```
- CMake Tools - ```ms-vscode.cmake-tools```

### Install Cuda on Ubuntu - only for PyCUDA!

NOTE: With ```cuda-toolkit``` from conda you won't be able to debug native CUDA code. When you will try to debug eg. [Cuda Samples](https://github.com/NVIDIA/cuda-samples) written in C/C++/CUDA the debugger won't run because ```cuda-gdb``` installed with conda is just a debugger selector. Use Nvidia system wide installer to get ```cuda-gdb``` running.

Look for latest cuda toolkit

```conda search -c nvidia cuda-toolkit```

As of 2025 H2, there is no compatible PyCuda package in conda that works with Cuda toolkit 13.0 on Python 3.11, so install older CUDA.

```conda install -c nvidia cuda-toolkit=12.8.1```

### Install PyCuda

Then use conda forge to build PyCuda

```conda install conda-forge::pycuda```

Run Nvidia-smi to check your driver. This tool also reports CUDA Version in the upper right corner of the output, but beware this is the latest supported CUDA version, not the installed CUDA version.

```nvidia-smi```

# Running CUDA Samples

Clone [Cuda Samples Repo](https://github.com/NVIDIA/cuda-samples).

Add the following in VS Code workspace ```settings.json```

```
    "cmake.cmakePath": "/usr/bin/cmake",
    "cmake.buildDirectory": "${workspaceFolder}/build",
    "cmake.sourceDirectory": "${workspaceFolder}",
    "cmake.configureOnOpen": true,
    "cmake.parallelJobs": 8, // Adapt to your machine
    "cmake.configureEnvironment": {
        "CUDACXX": "/usr/local/cuda-12.8/bin/nvcc",
        "PATH": "/usr/local/cuda-12.8/bin:${env:PATH}",
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.8/lib64:${env:LD_LIBRARY_PATH}"
    },
    "cmake.configureSettings": {
        "ENABLE_CUDA_DEBUG": "True",     // For CUDA kernel debugging
        "CMAKE_CUDA_ARCHITECTURES": "75" // Adapt to your GPUs
    }
```

Create configuration at ```launch.json```. Note that ```cuda-gdb``` needs to be used in order to debug GPU kernel code.

```
        {
            "name": "CUDA GDB clock",
            "type": "cuda-gdb",
            "request": "launch",
            "program": "${workspaceFolder}/build/Samples/0_Introduction/clock/clock",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}/build/Samples/0_Introduction/clock",
            //"miDebuggerPath": "/usr/local/cuda-12.8/bin/cuda-gdb"
        },

```

Now, you should be able to run and debug native CUDA code.

# Containers

You can also run the development environment in the container. This will save you a lot of time setting up the environment.

## Setting up

[Docker Desktop on Windows 11 - video](https://www.youtube.com/watch?v=t7mkHFOeMdA)

On Windows you need the following to be installed and setup:
- WSL2 - containers will use WSL Linux kernel
- Docker
    - Enable docker integration in WSL - Settings -> Resources -> WSL integration -> enable for your Linux distro
- VS Code with extensions: "Dev Containers"

In WSL install:
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Running CUDA container

This project is already setup to build a container needed to run the course. VS code should prompt you to open the folder in a container. If you want run this repo in the container manually run:

```Dev Containers: Reopen in Container```

When Dockerfile is modified and image did not get updated you may need to rebuild:

```Dev Containers: Rebuild and Reopen in Container```

## Setup in the container

Create a venv (yes, in the container)

```python3 -m venv .venv_container```

Then install packages from the ```requirements.txt``` file with ```pip```
