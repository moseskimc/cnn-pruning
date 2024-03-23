"""This module downloads FashionMNIST to local"""
import torch
from torchvision import transforms, datasets

# download FashionMNIST data
# train
fmnist_train = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=transforms.ToTensor()
)
# test
fmnist_test = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=transforms.ToTensor()
)


# get dataloaders
def get_dataloader(
    train: bool,
    batch_size: int,
):

    dataset = fmnist_train if train else fmnist_test

    return torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True
    )
