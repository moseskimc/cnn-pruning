# cnn-pruning

## Introduction

 This repository is a PyTorch implementation of the taylor expansion criterion found in [*Pruning Convolutional Neural Networks for Resource Efficient Inference*](https://arxiv.org/pdf/1611.06440.pdf), testing the performance of a simple CNN model with only 3 convolutional layers on FashionMNIST after pruning.


## How to run

### Local

#### Environment

First, create your environment including dependencies: `torchvision`, `torchinfo`, and `mlflow`. If using conda, install the dependencies via the command
    
    conda create --name <env> --file requirements.txt

Also, make sure to export the current working directory via the command `export PYTHONPATH=$(pwd)` so that all the modules are imported properly when running scripts. 

#### MLflow

In order to launch the MLflow tracking server, run the command below

    mlflow server --host 127.0.0.1 --port 5005

in a separate terminal window.

#### Train

Once MLflow server has started, train, prune, and fine-tune the model running the command below:

    python src/scripts/train.py

### Docker

If running on Apple silicon please export the following variable in your terminal before building your Docker image.

    export DOCKER_DEFAULT_PLATFORM=linux/amd64

1. Build the custom image
```
    docker build -t pytorch-jupyter . -f Dockerfile
```
2. Run the container

```
    docker run --ipc=host -ti --rm \
        -v $(pwd):/usr/app -p 8888:8888 -p 5001:5000 \
        --name pytorch-jupyter \
        pytorch-jupyter:latest
```

You will be able to access the jupyter server at `localhost:8888` on your web browser. In order to run `mlflow` in the same container execute the command below:

    docker exec -ti pytorch-jupyter mlflow ui --port 5000 --host 0.0.0.0

Now that you have `mlflow` running, you are ready to run the scripts inside the container.

#### Train

The following command first trains the model and prunes it with a calibration/fine-tuning step at the end. If you wish to change some of the training or pruning params, please edit the corresponding files in `src/config/`.

    docker exec -ti pytorch-jupyter python src/scripts/train.py

#### Inference


    docker exec -ti pytorch-jupyter python src/scripts/predict.py

## Results

Due to lack of computing rsources, a small CNN with 3 convolutional layers (correponding to layer 0, layer 2, and layer 4) was pruned by layer as follows:

- Layer 0: 50.0%  (1 out of 2)
- Layer 2: 50.0%  (2 out of 4)
- Layer 4: 62.5%  (5 out of 8)


Model summaries, detailing the number of parameters per layer, before and after pruning are shown below.

### Before pruning

    =================================================================
    Layer (type:depth-idx)                   Param #
    =================================================================
    CNN                                      --
    ├─Sequential: 1-1                        --
    │    └─Conv2d: 2-1                       20
    │    └─ReLU: 2-2                         --
    │    └─Conv2d: 2-3                       76
    │    └─ReLU: 2-4                         --
    │    └─Conv2d: 2-5                       296
    │    └─ReLU: 2-6                         --
    │    └─Flatten: 2-7                      --
    │    └─Linear: 2-8                       62,730
    =================================================================
    Total params: 63,122
    Trainable params: 63,122
    Non-trainable params: 0
    =================================================================

### After pruning

    =================================================================
    Layer (type:depth-idx)                   Param #
    =================================================================
    CNN                                      --
    ├─Sequential: 1-1                        --
    │    └─Conv2d: 2-1                       10
    │    └─ReLU: 2-2                         --
    │    └─Conv2d: 2-3                       20
    │    └─ReLU: 2-4                         --
    │    └─Conv2d: 2-5                       57
    │    └─ReLU: 2-6                         --
    │    └─Flatten: 2-7                      --
    │    └─Linear: 2-8                       23,530
    =================================================================
    Total params: 23,617
    Trainable params: 23,617
    Non-trainable params: 0
    =================================================================

### Metrics

<img src="resources/metrics.png">

The original model was trained for 10 epochs and the pruned model was calibrated/fine-tuned for another 10 epochs. Observe how the performance of the pruned model at the 6th epoch achieves an accuracy similar to that of the original model even with more than half of the parameters pruned.
