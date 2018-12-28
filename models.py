import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, Linear, ReLU, Dropout, Sequential


class MLP(Block):
    """
    A simple multilayer perceptron model based on our custom Blocks.
    Architecture is:

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.

    If dropout is used, a dropout layer is added after every ReLU.
    """
    def __init__(self, in_features, num_classes, hidden_features=(),
                 dropout=0, **kw):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param: Dropout probability. Zero means no dropout.
        """
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_features = hidden_features
        self.dropout = dropout

        blocks = []

        # TODO: Build the MLP architecture as described.
        # ====== YOUR CODE: ======

        # First layer

        # trying without **kw
        blocks.append(Linear(in_features, hidden_features[0], **kw))
        blocks.append(ReLU())
        blocks.append(Dropout(self.dropout)) if self.dropout != 0 else None

        num_of_layers = len(hidden_features) - 1
        # Iterate over hidden layers only
        for i in range(num_of_layers):
            blocks.append(Linear(self.hidden_features[i], self.hidden_features[i+1], **kw))
            blocks.append(ReLU())
            blocks.append(Dropout(self.dropout)) if self.dropout != 0 else None
            if i+1 == num_of_layers:
                break

        # Last layer
        blocks.append(Linear(self.hidden_features[-1], self.num_classes, **kw))

        # ========================

        self.sequence = Sequential(*blocks)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f'MLP, {self.sequence}'


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======

        conv_size = 3 #by 3
        max_pool_size = 2 #by 2

        for i, filter in enumerate(self.filters):
            layers.append(nn.Conv2d(in_channels, filter, conv_size, padding=1))
            in_channels = filter
            layers.append(nn.ReLU())
            if (i+1) % self.pool_every == 0:
                if in_h >= max_pool_size and in_w >= max_pool_size:
                    layers.append(nn.MaxPool2d(max_pool_size))
                    in_h = (in_h - 2) / 2 + 1
                    in_w = (in_w - 2) / 2 + 1

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        num_of_pools = int(len(self.filters) / self.pool_every)
        print("num of pools: ", num_of_pools)
        for i in range(num_of_pools):
            if in_w == 1 or in_h == 1:
                break
            in_h = (in_h - 2) / 2 + 1
            in_w = (in_w - 2) / 2 + 1


        in_features = self.filters[-1] * in_w * in_h
        print("in_h: {}, in_w: {}, in_features: {}".format(in_h, in_w, in_features))
        for layer in self.hidden_dims:
            layers.append(nn.Linear(int(in_features), layer))
            layers.append(nn.ReLU())
            in_features = layer
        layers.append(nn.Linear(in_features, self.out_classes))

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======

        features = self.feature_extractor(x)
        features = features.reshape(features.shape[0], -1)
        out = self.classifier(features)

        # ========================
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.
    # ====== YOUR CODE: ======

    new_size = lambda conv_result: conv_result if conv_result >= 1 else 1
    conv_new_size = lambda size_in, padding, kernel_size, stride: (size_in + 2 * padding - (kernel_size - 1) - 1) / 2

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        self.in_h = in_h
        self.in_w = in_w
        layers = []

        conv_size = 3 #by 3
        avg_pool_size = 4 #by 2

        # stride 2 stride 1 same out as in
        layers.append(nn.Conv2d(in_channels, 32, conv_size, stride=1, padding=1, bias=False))
        in_channels = 32
        layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nn.ReLU())

        for i, filter in enumerate(self.filters):

            # stride 2 stride 1 same out as in
            layers.append(nn.Conv2d(in_channels, in_channels, conv_size, stride=1, groups=in_channels, padding=1, bias=False))
            self.in_h = int((self.in_h + 2 - (conv_size - 1) - 1) / 1) + 1
            self.in_w = int((self.in_w + 2 - (conv_size - 1) - 1) / 1) + 1

            layers.append(nn.BatchNorm2d(in_channels))
            layers.append(nn.ReLU())

            layers.append(nn.Conv2d(in_channels, filter, 1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(filter))
            layers.append(nn.ReLU())

            in_channels = filter

        if self.in_h >= 2 and self.in_w >= 2:
            layers.append(nn.AvgPool2d(avg_pool_size))
            self.in_h = int((self.in_h - 2) / avg_pool_size) + 1
            self.in_w = int((self.in_w - 2) / avg_pool_size) + 1

        if self.in_h < 1:
            self.in_h = 1
        if self.in_w < 1:
            self.in_w = 1

        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):

        layers = []

        in_features = self.filters[-1] * self.in_w * self.in_h
        for layer in self.hidden_dims:
            layers.append(nn.Linear(int(in_features), layer))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout())
            in_features = layer
        layers.append(nn.Linear(in_features, self.out_classes))

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    # ========================

