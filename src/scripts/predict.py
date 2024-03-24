import mlflow
import torch

from src.config.inference_params import RUN_ID, FMNIST_LABELS
from src.data.fashion_mnist import fmnist_test


if __name__ == "__main__":
    # Connect to the MLflow server
    mlflow.set_tracking_uri("http://0.0.0.0:5000")
    # Load model
    model_path = f"runs:/{RUN_ID}/model_pruned"
    loaded_model = mlflow.pyfunc.load_model(model_path)

    # Load sample data (first image from batch)
    image_sample, label = fmnist_test[0][0][None, :], fmnist_test[0][1]
    print(f"Ground truth label is: {label}")

    # Predict
    outputs = loaded_model.predict(image_sample.numpy())
    outputs_tensor = torch.from_numpy(outputs)
    argmax = torch.argmax(outputs_tensor).item()
    predicted_label = FMNIST_LABELS[argmax]
    print(f"Predicted index is: {argmax}")
    print(f"Predicted label is: {predicted_label}")
