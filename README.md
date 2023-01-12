# Network Intrusion Detection with Deep Learning

Network intrusions and computer attacks can have severe consequences for both businesses and individuals. As cybercrime becomes more advanced, traditional signature-based detection methods are becoming less effective. To combat this, Deep Learning (DL) has emerged as a powerful technology for addressing various cybersecurity challenges. 

This repository contains a series of labs that guide students and practitioners through the hands-on implementation of DL-based systems for network intrusion and anomaly detection. It covers topics such as the analysis of malicious network traffic, the design and optimization of neural network models, and the practical challenges of deploying these systems in real-world settings.

## Table of Contents
1. Simple test of DDoS attack detection with a convolutional neural network


## Requirements

These laboratories have been implemented as Jupyter notebooks and stand-alone Python scripts, using Python v3.9, Keras and Tensorflow 2, Numpy and Pyshark. 

The installation of all the necessary tools and libraries can be done by using the ```conda``` software environment (https://docs.conda.io/projects/conda/en/latest/).
We suggest the installation of ```miniconda```, a light version of ```conda```. ```miniconda``` is available for MS Windows, MacOSX and Linux and can be installed by following the guidelines available at https://docs.conda.io/en/latest/miniconda.html#. 

For instance, in Linux OS, execute the following command and follows the on-screen instructions:

```
bash Miniconda3-latest-Linux-x86_64.sh
```

Similarly, in Mac OS:

```
bash Miniconda3-latest-MacOSX-arm64.sh
```

Then create a new ```conda``` environment (called ```python39```) based on Python 3.9:

```
conda create -n phdcourse python=3.9
```

Activate the new ```phdcourse ``` environment:

```
conda activate phdcourse
```

Install Tensorflow 2.7.0. In a Linux OS:

```
(phdcourse)$ pip3 install tensorflow 
```

On the new Apple computers with M1 CPU, execute instead:

```
(phdcourse)$ conda install -c apple tensorflow-deps
(phdcourse)$ pip3 install tensorflow-macos 
(phdcourse)$ pip3 install tensorflow-metal 
```

And finalise the installation with a few more packages:

```
(phdcourse)$ pip3 install pyshark scikit-learn numpy matplotlib h5py lxml jupyter
```


Pyshark is just a Python wrapper for tshark, allowing python packet parsing using Wireshark dissectors. This means that ```tshark``` must be also installed. On an Ubuntu-based OS, use the following command:

```
sudo apt install tshark
```

### Additional tools
1. [Pycharm Community IDE](https://www.jetbrains.com/pycharm/download/): a Python development environment with code auto-completion and debugging tools. 