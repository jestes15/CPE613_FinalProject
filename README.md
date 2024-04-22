# CPE 613 Semester Project

## Prerequisites

> :warning: **MUST BE ON X86-64 SYSTEM**: Intel MKL only supports x86-64 systems, ARM systems will need modifications to compile and run.

> :warning: **Users of Windows Systems**: This code has not been tested in a Windows enviroment, only in a Linux enviroment.

### CUDA
I am currently using CUDA version 12.4 and an RTX 4070 Ti.

1. To install cuda, navigate to [this](https://developer.nvidia.com/cuda-downloads) page and download CUDA for your operating system.
    1. Follow the instructions on the website, I will show how to install in WSL.
2. Run 
```bash
$ wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
$ sudo dpkg -i cuda-keyring_1.1-1_all.deb
$ sudo apt-get update
$ sudo apt-get -y install cuda-toolkit-12-4
```
3. 