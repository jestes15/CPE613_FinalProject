# CPE 613 Semester Project

## Prerequisites

> :warning: **MUST BE ON X86-64 SYSTEM**: Intel MKL only supports x86-64 systems, ARM systems will need modifications to compile and run.

> :warning: **Users of Windows Systems**: This code has not been tested in a Windows enviroment, only in a Linux enviroment.

### NVIDIA CUDA
I am currently using CUDA version 12.4 and an RTX 4070 Ti.

1. To install CUDA, navigate to [this](https://developer.nvidia.com/cuda-downloads) page and download CUDA for your operating system.
    1. Follow the instructions on the website, I will show how to install in WSL.
2. Run the following commands to install CUDA on Ubuntu for WSL:
```bash
$ wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
$ sudo dpkg -i cuda-keyring_1.1-1_all.deb
$ sudo apt-get update
$ sudo apt-get -y install cuda-toolkit-12-4
```
3. Verify CUDA is installed by running `nvcc --version` and the output should be:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Feb_27_16:19:38_PST_2024
Cuda compilation tools, release 12.4, V12.4.99
Build cuda_12.4.r12.4/compiler.33961263_0
```

### Intel oneAPI
1. To install oneAPI, navigate to [this](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) page and download oneAPI for your operating system.
    1. Follow the instructions on the website, I will show how to install in WSL.
2. Run the following commands to install oneAPI on Ubuntu for WSL:
```bash
$ sudo apt update
$ sudo apt install -y gpg-agent wget
$ wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
$ echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
$ sudo apt update
$ sudo apt install intel-basekit intel-hpckit
```
3. Verify CUDA is installed by running `source /opt/intel/oneapi/setvars.sh` and the output should be:
```
:: initializing oneAPI environment ...
   -bash: BASH_VERSION = 5.1.16(1)-release
   args: Using "$@" for setvars.sh arguments:
:: advisor -- latest
:: ccl -- latest
:: compiler -- latest
:: dal -- latest
:: debugger -- latest
:: dev-utilities -- latest
:: dnnl -- latest
:: dpcpp-ct -- latest
:: dpl -- latest
:: ipp -- latest
:: ippcp -- latest
:: mkl -- latest
:: mpi -- latest
:: tbb -- latest
:: vtune -- latest
:: oneAPI environment initialized ::
```

### Cython
1. In the FFT_IFFT directory, run:
```bash
$ pip install -r requirements.txt
```

## Building and Running
### Building
To build the library, execute the makeRelease.sh script.

### Running
1. To execute Python code using this library, the oneAPI enviroment variables must be set. To set these variables, run `source /opt/intel/oneapi/setvars.sh` and verify that the following was output:
```
:: initializing oneAPI environment ...
   -bash: BASH_VERSION = 5.1.16(1)-release
   args: Using "$@" for setvars.sh arguments:
:: advisor -- latest
:: ccl -- latest
:: compiler -- latest
:: dal -- latest
:: debugger -- latest
:: dev-utilities -- latest
:: dnnl -- latest
:: dpcpp-ct -- latest
:: dpl -- latest
:: ipp -- latest
:: ippcp -- latest
:: mkl -- latest
:: mpi -- latest
:: tbb -- latest
:: vtune -- latest
:: oneAPI environment initialized ::
```

2. After these are set, just run `python3 test_fft.py`