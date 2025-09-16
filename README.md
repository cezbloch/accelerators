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


### Install Cuda on Ubuntu

Look for latest cuda toolkit

```conda search -c nvidia cuda-toolkit```

As of 2025 H2, there is no compatible PyCuda package in conda that works with Cuda toolkit 13.0 on Python 3.11, so install older CUDA.

```conda install -c nvidia cuda-toolkit=12.8.1```

### Install PyCuda

Then use conda forge to build PyCuda

```conda install conda-forge::pycuda```

Run Nvidia-smi to check your driver. This tool also reports CUDA Version in the upper right corner of the output, but beware this is the latest supported CUDA version, not the installed CUDA version.

```nvidia-smi```