import torch

import torch.nn.functional as F

import numpy as np

from collections import defaultdict
from operator import itemgetter
from heapq import nsmallest


from src.utils import to_device, get_default_device


# Create a class to keep track of filter rankings
# when performing a backward pass gradient computation
class FilterRanker:

    def __init__(self, model, loss_func):
        self.model = model
        # get all layers from nn.Sequential 'network'
        self.model_layers = model._modules["network"]._modules.items()
        # loss function
        self.loss_func = loss_func
        # reset rankings data when instantiating
        self.initialize_filter_rankings()

        # when performing the backward pass
        # we keep track of which layer we are computing the gradient of
        self.conv_layer_grad_index = 0

    def initialize_filter_rankings(self):
        # for each convolutional layer
        # we keep track of the filter rankings
        self.layer_filter_rankings = {}

    def normalize_rankings_per_layer(self):
        for layer in self.layer_filter_rankings:
            layer_filters = self.layer_filter_rankings[layer]
            f_i = torch.abs(layer_filters).cpu()
            # we normalize each layer tensor
            f_i = f_i / np.sqrt(torch.sum(f_i * f_i))
            self.layer_filter_rankings[layer] = f_i

    # use minheap to get lowest k ranking filters
    def lowest_ranking_filters(self, k):
        """Returns lowest overall k filters by ranking"""
        data = []

        # layers here are just offsets from the
        # list conv layers (i.e. 0, 1, 2, ..., no_layers)
        lfr = self.layer_filter_rankings

        for layer in sorted(lfr.keys()):
            for filter_idx in range(lfr[layer].size(0)):
                # we need the layer index from the original model
                model_layer = self.conv_layer_indices[layer]
                # the filter ranking for the current index
                filter_idx_ranking = lfr[layer][filter_idx]
                # append
                data.append((model_layer, filter_idx, filter_idx_ranking))

        return nsmallest(k, data, itemgetter(2))

    # function to pass as hook when gradients are computed
    def get_filter_rank(self, grad):
        # since we compute grads via backpropagation
        # the gradients of the last layer are computed first
        # forward pass is performed first
        # note: this function won't be called until gradients are computed

        # we access the activation tensors from the last conv and go backwards
        activation_index = len(self.activations) - self.conv_layer_grad_index - 1
        # access the activation
        activation = self.activations[activation_index]

        # compute the ranking
        taylor_exp = torch.abs(
            activation * grad
        )  # as a result of taylor expansion (ref to paper)
        filter_rankings = taylor_exp.mean(
            dim=(0, 2, 3)
        ).data  # average over all other indices

        # check if any filter rankings have been added to the activation index
        device = get_default_device()
        if activation_index not in self.layer_filter_rankings:
            self.layer_filter_rankings[activation_index] = to_device(
                torch.FloatTensor(activation.size(1)).zero_(), device
            )

        self.layer_filter_rankings[
            activation_index
        ] += filter_rankings  # for each batch
        self.conv_layer_grad_index += 1  # must be reset for each batch

    def forward(self, x):

        # reset layer index moving backwards
        self.conv_layer_grad_index = 0

        # keep track of activations in feedforward order
        self.activations = []
        self.grad_layer_index = 0  # conv layer index
        # starting from last

        # dictionary to keep the order of conv layers
        # recall there are other layers between conv layers
        self.conv_layer_indices = {}

        # for a single forward pass
        conv_layer_ct = 0
        for layer, (name, module) in enumerate(self.model_layers):
            # get the activation by passing to the module
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                # register hook
                x.register_hook(self.get_filter_rank)
                # activations
                self.activations.append(x)
                # conv layer indices (i.e. model layer index)
                self.conv_layer_indices[conv_layer_ct] = layer
                conv_layer_ct += 1

        return x


class Pruner(FilterRanker):
    def __init__(self, model):
        super().__init__(model, loss_func=F.cross_entropy)

    def get_filter_rankings(self, train_loader):
        for batch in train_loader:
            images, labels = batch
            output = self.forward(images)
            self.loss_func(output, labels).backward()

    def prune_model(
        self,
        train_loader,
        k_filters,
    ):

        self.get_filter_rankings(train_loader)  # get rankings for one epoch
        self.normalize_rankings_per_layer()  # normalize rankings

        lowest_k_filters = self.lowest_ranking_filters(k_filters)

        layer_filter_pairs = self.adjust_subseq_layer_kernel_pos(lowest_k_filters)

        for layer, f_id in layer_filter_pairs:
            # prune layer
            self.prune_layer(layer, f_id)

        return self.model

    def adjust_subseq_layer_kernel_pos(self, min_heap_res):

        filter_idx_by_layer = defaultdict(list)

        for layer, f_idx, _ in min_heap_res:
            filter_idx_by_layer[layer].append(f_idx)

        # here's the catch!
        # we order the filter indices in increasing order
        # every subsequent filter removal will result
        # in the corresponding kernel in the next layer to be removed
        # the first removal will correspond to the kernel index
        # but, the second removal will correspond to the kernel index - 1
        # the third removal will corerspond to the kernel index - 2
        # and so forth;

        # similarly, the ordering of the filters in the current layer changes
        # in the same manner
        adjusted_dict = {}
        for layer in filter_idx_by_layer:
            filter_indices = filter_idx_by_layer[layer]

            filter_indices_sorted = sorted(filter_indices)

            adjusted_indices = []
            for i, f_i in enumerate(filter_indices_sorted):
                adjusted_indices.append(f_i - i)

            adjusted_dict[layer] = adjusted_indices

        # finally, output all layer, f_idx pairs
        layer_filter_pairs = []

        for layer in adjusted_dict:
            for f_idx in adjusted_dict[layer]:
                layer_filter_pairs.append((layer, f_idx))

        return layer_filter_pairs

    def prune_layer(self, layer, f_idx):

        model = self.model
        conv_layer_indices = self.conv_layer_indices

        # ordered conv_layers (feedforward direction)

        model_layers = model._modules["network"]._modules.items()
        conv_layers = list(model_layers)
        _, conv = conv_layers[layer]

        # remove filter from conv layer
        # instantiate layer
        new_conv = torch.nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels - 1,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
        )
        # assign weights
        conv_old_weights = conv.weight.data.cpu().numpy()
        new_conv_weights = new_conv.weight.data.cpu().numpy()

        # remove filter
        new_conv_weights[:f_idx, :, :, :] = conv_old_weights[:f_idx, :, :, :]
        new_conv_weights[f_idx:, :, :, :] = conv_old_weights[f_idx + 1:, :, :, :]
        new_conv.weight.data = torch.from_numpy(new_conv_weights)
        # set to cuda
        if torch.cuda.is_available():
            new_conv.weight.data = new_conv.weight.data.cuda()

        # remove corresponding bias dimension
        bias_numpy = conv.bias.data.cpu().numpy()
        bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
        bias[:f_idx] = bias_numpy[:f_idx]
        bias[f_idx:] = bias_numpy[f_idx + 1:]
        new_conv.bias.data = torch.from_numpy(bias)
        # set to cuda
        if torch.cuda.is_available():
            new_conv.bias.data = new_conv.bias.data.cuda()

        # replace old layer with new
        model._modules["network"][layer] = new_conv

        # remove kernel index in subsequent layer (if applicable)
        # find whether there is a subsequent conv layer
        conv_layer_index = [
            key for key, val in conv_layer_indices.items() if val == layer
        ][0]

        next_index = conv_layer_index + 1

        if next_index in conv_layer_indices:

            # get original model layer index
            orig_layer_idx = conv_layer_indices[next_index]

            # get next conv
            _, next_conv = conv_layers[orig_layer_idx]

            # remove corresponding kernel index in all filters of this layer
            new_next_conv = torch.nn.Conv2d(
                in_channels=next_conv.in_channels - 1,
                out_channels=next_conv.out_channels,
                kernel_size=next_conv.kernel_size,
                stride=next_conv.stride,
                padding=next_conv.padding,
                dilation=next_conv.dilation,
                groups=next_conv.groups,
                bias=(next_conv.bias is not None),
            )

            # assign weights
            next_conv_old_weights = next_conv.weight.data.cpu().numpy()
            new_next_conv_weights = new_next_conv.weight.data.cpu().numpy()

            new_next_conv_weights[:, :f_idx, :, :] = next_conv_old_weights[
                :, :f_idx, :, :
            ]
            new_next_conv_weights[:, f_idx:, :, :] = next_conv_old_weights[
                :, f_idx + 1:, :, :
            ]

            new_next_conv.weight.data = torch.from_numpy(new_next_conv_weights)
            # assign to cuda
            if torch.cuda.is_available():
                new_next_conv.weight.data = new_next_conv.weight.data.cuda()

            # bias remains the same since we are not moving filters here
            new_next_conv.bias.data = next_conv.bias.data

            # replace old next layer with new next layer
            model._modules["network"][orig_layer_idx] = new_next_conv
        # need to prune the immediate linear layer after the last conv layer
        else:
            old_linear_layer = None
            linear_layer_idx = 0
            for layer, (name, module) in enumerate(
                model._modules["network"]._modules.items()
            ):
                if isinstance(module, torch.nn.Linear):
                    old_linear_layer = module
                    linear_layer_idx = layer
                    break

            # if we remove one filter we remove this many parameters
            params_per_ch = old_linear_layer.in_features // conv.out_channels

            new_linear_layer = torch.nn.Linear(
                old_linear_layer.in_features - params_per_ch,
                old_linear_layer.out_features,
            )

            old_linear_weights = old_linear_layer.weight.data.cpu().numpy()
            new_linear_weights = new_linear_layer.weight.data.cpu().numpy()

            new_linear_weights[:, : f_idx * params_per_ch] = old_linear_weights[:, : f_idx * params_per_ch]

            new_linear_weights[:, f_idx * params_per_ch:] = old_linear_weights[
                :, (f_idx + 1) * params_per_ch:
            ]

            new_linear_layer.bias.data = old_linear_layer.bias.data

            new_linear_layer.weight.data = torch.from_numpy(new_linear_weights)
            # assign to cuda
            if torch.cuda.is_available():
                new_linear_layer.weight.data = new_linear_layer.weight.data.cuda()

            # finally, replace layer
            # replace old next layer with new next layer
            model._modules["network"][linear_layer_idx] = new_linear_layer
