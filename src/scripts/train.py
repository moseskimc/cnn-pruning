from collections import defaultdict

import os
import copy
import torch
import torch.nn.functional as F
from torchinfo import summary

from src.modeling.baseline import CNN
from src.modeling.pruner import Pruner, FilterRanker
from src.data.dataloader import DeviceDataLoader
from src.data.fashion_mnist import get_dataloader
from src.utils import (
    get_default_device,
    to_device,
    train_model,
    evaluate_model,
)
from src.config.train_params import (
    NUM_EPOCHS,
    OPT_FUNC,
    LEARNING_RATE,
    BATCH_SIZE,
    DOCKER_SCRIPT,
)
from src.config.prune_params import K_FILTERS
from src.config.output import SAVE_DIR

import mlflow


if __name__ == "__main__":
    # Set tracking URI
    if DOCKER_SCRIPT:
        # Docker container
        mlflow.set_tracking_uri("http://0.0.0.0:5000")
    else:
        # Local
        mlflow.set_tracking_uri("http://127.0.0.1:5005")
    # name experiment
    mlflow.set_experiment("/cnn-pruning")

    # load train and test data
    train_dataloader = get_dataloader(train=True, batch_size=BATCH_SIZE)
    test_dataloader = get_dataloader(train=False, batch_size=BATCH_SIZE)

    # check if device is cuda
    device = get_default_device()

    train_loader = DeviceDataLoader(train_dataloader, device)
    test_loader = DeviceDataLoader(test_dataloader, device)

    with mlflow.start_run() as run:

        print(f"RUN_ID: {run.info.run_id}")
        print()

        params = {
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
        }
        # Log training parameters.
        mlflow.log_params(params)

        # instantiate model
        print("Model before prunning")
        model = CNN()
        model = to_device(model, device)

        # create save dir if it doesn't exist
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        # Log model summary.
        file_save_path = f"{SAVE_DIR}/model_summary.txt"
        with open(file_save_path, "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact(file_save_path)

        history = train_model(NUM_EPOCHS, LEARNING_RATE, model,
                              train_loader, test_loader, OPT_FUNC)
        # log train and val metrics
        for epoch, epoch_results in enumerate(history):
            val_loss = epoch_results["val_loss"]
            mlflow.log_metric("val_loss", f"{val_loss:2f}", step=epoch)
            train_loss = epoch_results["train_loss"]
            mlflow.log_metric("train_loss", f"{train_loss:2f}", step=epoch)
            val_acc = epoch_results["val_acc"]
            mlflow.log_metric("val_accuracy", f"{val_acc:2f}", step=epoch)
        # Save the trained model to MLflow.
        print("Saving model...")
        mlflow.pytorch.log_model(model, "model")
        print()
        print("Pruning model...")
        # compute rankings
        loss_func = F.cross_entropy
        fr = FilterRanker(model, loss_func)
        # run one epoch
        for batch in train_loader:
            images, labels = batch
            output = fr.forward(images)
            # everytime we perform a backward pass
            # we add to layer_filter_rankings
            fr.loss_func(output, labels).backward()

        fr.normalize_rankings_per_layer()  # finally, we normalize per layer

        # Get lowest ranked k filters by ranking.
        # Note that these filters are ranked across all layers.
        lowest_filters = fr.lowest_ranking_filters(K_FILTERS)
        layer_filters_pruned = defaultdict(int)
        for layer, filter_index, _ in lowest_filters:
            layer_filters_pruned[layer] += 1

        print("Percentage of filters to be pruned")
        for layer in layer_filters_pruned:
            no_filters_to_prune = layer_filters_pruned[layer]
            total_filters = model._modules["network"][layer].out_channels
            prune_perc_layer = round(
                100
                * no_filters_to_prune
                / total_filters
                * 1.0,
                2,
            )
            print(f"Layer {layer}: {prune_perc_layer}%",
                  f" ({no_filters_to_prune} out of {total_filters})")

        # prune model
        # deep copy model
        model_copy = copy.deepcopy(model)

        # instantiate Pruner
        pr = Pruner(model_copy)

        model_pruned = pr.prune_model(train_loader,
                                      k_filters=K_FILTERS)

        # save pruned model summary as artifact in MLflow
        # Log model summary.
        file_save_path = f"{SAVE_DIR}/pruned_model_summary.txt"
        with open(file_save_path, "w") as f:
            f.write(str(summary(model_pruned)))
        mlflow.log_artifact(file_save_path)
        # check number of filters after pruning
        for layer, (name, module) in enumerate(
            model_pruned._modules["network"]._modules.items()
        ):
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                layer_filters = module.out_channels
                print(f"Layer {layer} has: {layer_filters}")

        # print initial params
        evaluate_model(model_pruned, test_loader)

        # fine-tune the prune model
        print("Fine-tuning model...")
        history = train_model(
            NUM_EPOCHS, LEARNING_RATE, model_pruned,
            train_loader, test_loader, OPT_FUNC
        )
        # log train and val metrics for pruned model during fine-tuning
        for epoch, epoch_results in enumerate(history):
            val_loss = epoch_results["val_loss"]
            mlflow.log_metric("val_loss-pruned_model", f"{val_loss:2f}",
                              step=epoch)
            train_loss = epoch_results["train_loss"]
            mlflow.log_metric("train_loss-pruned_model", f"{train_loss:2f}",
                              step=epoch)
            val_acc = epoch_results["val_acc"]
            mlflow.log_metric("val_accuracy-pruned_model", f"{val_acc:2f}",
                              step=epoch)

        # Save the trained model to MLflow.
        print("Saving pruned model...")
        mlflow.pytorch.log_model(model_pruned, "model_pruned")
