FROM cschranz/gpu-jupyter:v1.7_cuda-12.2_ubuntu-22.04

USER root

RUN apt-get update
RUN apt install -y swig build-essential python3-dev 

RUN pip install --upgrade pip
RUN pip install "gymnasium[box2d]" opencv-python IPython pygame

RUN fix-permissions "/home/jovyan"