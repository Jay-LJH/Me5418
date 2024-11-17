FROM jupyter/tensorflow-notebook

USER root

RUN apt-get update
RUN apt install -y imagemagick

RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install gymnasium opencv-python IPython pygame
RUN pip install swig
RUN pip install "gymnasium[box2d]"
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN fix-permissions "/home/jovyan"