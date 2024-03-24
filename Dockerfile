FROM ubuntu:latest
FROM pytorch/pytorch:latest

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
# Dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        tini \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0

WORKDIR /usr/app

RUN python -m pip install --upgrade pip
# Jupyter
RUN pip install jupyter
# fastdup https://github.com/visual-layer/fastdup
RUN pip install fastdup
RUN pip install opencv-python
RUN pip install matplotlib matplotlib-inline pandas
RUN pip install pillow
RUN pip install pyyaml
RUN pip install torchinfo
# YOLO 8.1
RUN pip install ultralytics Cython>=0.29.32 lapx>=0.5.5
# MLFlow 2.10
RUN pip install mlflow pytorch_lightning

ENV PYTHONPATH /usr/app

COPY . . 

# Start Jupyter server
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]