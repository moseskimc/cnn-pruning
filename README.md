# cnn-pruning

## Introduction

 This repository is a PyTorch implementation of the taylor expansion criterion found in [*Pruning Convolutional Neural Networks for Resource Efficient Inference*](https://arxiv.org/pdf/1611.06440.pdf), testing the performance of a simple CNN model with only 2 convolutional layers on FashionMNIST after pruning.


## MLflow

In order to launch the MLflow tracking server, run the command below

    mlflow server --host 127.0.0.1 --port 5000

in a separate terminal window.

## Train

Once you the server has started, train, prune, and fine-tune the model running the commands below:

    export PYTHONPATH=$(pwd)
    python src/scripts/train.py
