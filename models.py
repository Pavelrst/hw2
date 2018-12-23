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
        in_block = in_features
        for key, value in kw.items():
            print(key)
            print (value)
        for i, h in enumerate(self.hidden_features):
            #print('  in_bloc=',in_block, 'h=',h)
            blocks.append(Linear(in_block,h,**kw))
            blocks.append(ReLU())
            in_block = h
        blocks.append(Linear(in_block,num_classes,value))
        

        
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
        :param filters: .A list of of length N containing the number of
            filters in each conv layer
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

        #TODO: pool_every = P = the number of conv layers before each max-pool
        N = len(self.filters) # num of conv layers in total.
        P = self.pool_every # num of conv layers before each pooling.
        N_P = int(N/P) # num of (Conv -> ReLU)*P -> MaxPool in total.

        #print("N(num of conv layers in total)=",N)
        # print("P(num of conv layers before each pooling.)=",P)
        # print("N_P(num of (Conv -> ReLU)*P -> MaxPool)=",N_P)
        # TODO: iterate over pooling layers
        # TODO: for each pool, iterate over conv layers for this pool
        # TODO: add all those layers to layers list.

        #print("input shape(Ch,H,W) = (",in_channels,",",in_h,",",in_w,")")

        # save all dimensions for further usage
        curr_H_in = in_h
        curr_W_in = in_w
        curr_dim = (in_channels, in_h, in_w)
        dims_list = []
        dims_list.append(curr_dim)

        filters_list_idx = 0
        curr_in_channels = in_channels
        for pool_idx in range(N_P):
            for conv_idx in range(P):
                kernel_size = 3 # 3x3
                stride = 1
                padding = 1
                dilation = 1

                # calc dimension of output tensor
                out_channels = self.filters[filters_list_idx] # as num of filters
                H_out = int((curr_H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
                W_out = int((curr_W_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
                curr_dim = (out_channels,H_out,W_out)
                dims_list.append(curr_dim)
                curr_H_in = H_out
                curr_W_in = W_out

                # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
                conv = nn.Conv2d(curr_in_channels,out_channels,kernel_size,stride,padding)
                #print("adding conv layer. in channels = ",curr_in_channels," out dim = ",curr_dim)
                curr_in_channels = out_channels
                layers.append(conv)
                #print("adding RELU layer")
                layers.append(nn.ReLU())
                filters_list_idx+=1

            H_out = int(H_out/2)
            W_out = int(W_out/2)
            curr_dim = (out_channels, H_out, W_out)
            dims_list.append(curr_dim)
            curr_H_in = H_out
            curr_W_in = W_out
            #print("adding Maxpool layer. out dim = ",curr_dim)
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            layers.append(pool)
        self.dim_list = dims_list

            

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

        #TODO: calc in features
        ch,h,w = self.dim_list[-1]
        in_features = ch*h*w
        #print("in fetures of classifier is",in_features)

        layers.append(nn.Linear(in_features, self.hidden_dims[0]))
        layers.append(nn.ReLU())
        for curr_dim_idx in range(len(self.hidden_dims)-1):
            layers.append(nn.Linear(self.hidden_dims[curr_dim_idx], self.hidden_dims[curr_dim_idx+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        #print("x shape =",x.size())
        features = self.feature_extractor(x)
        #print("features shape =", features.size())
        features = features.view(features.size(0), -1)
        #print("features shape =", features.size())
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
    #raise NotImplementedError()
    # ========================

