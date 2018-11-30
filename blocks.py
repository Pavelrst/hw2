import abc
import torch


class Block(abc.ABC):
    """
    A block is some computation element in a network architecture which
    supports automatic differentiation using forward and backward functions.
    """
    def __init__(self):
        # Store intermediate values needed to compute gradients in this hash
        self.grad_cache = {}
        self.training_mode = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Computes the forward pass of the block.
        :param args: The computation arguments (implementation specific).
        :return: The result of the computation.
        """
        pass

    @abc.abstractmethod
    def backward(self, dout):
        """
        Computes the backward pass of the block, i.e. the gradient
        calculation of the final network output with respect to each of the
        parameters of the forward function.
        :param dout: The gradient of the network with respect to the
        output of this block.
        :return: A tuple with the same number of elements as the parameters of
        the forward function. Each element will be the gradient of the
        network output with respect to that parameter.
        """
        pass

    @abc.abstractmethod
    def params(self):
        """
        :return: Block's trainable parameters and their gradients as a list
        of tuples, each tuple containing a tensor and it's corresponding
        gradient tensor.
        """
        pass

    def train(self, training_mode=True):
        """
        Changes the mode of this block between training and evaluation (test)
        mode. Some blocks have different behaviour depending on mode.
        :param training_mode: True: set the model in training mode. False: set
        evaluation mode.
        """
        self.training_mode = training_mode


class Linear(Block):
    """
    Fully-connected linear layer.
    """

    def __init__(self, in_features, out_features, wstd=0.1):
        """
        :param in_features: Number of input features (Din)
        :param out_features: Number of output features (Dout)
        :wstd: standard deviation of the initial weights matrix
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # TODO: Create the weight matrix (w) and bias vector (b).

        # ====== YOUR CODE: ======
        """
        Pavel done this:
        W = out_features x in_features matrix
        b = out_features column vector
        y=f(xW_T + b) -> which is what?
        """

        #   Create normal distribution sampler. We will fill our tensors.
        #   from this distribution.
        norm_dist = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([wstd]))

        #   Create the w, b tensors of given sizes.
        #   [:,:,0] is for disabling additional dimension which is 1
        self.w = norm_dist.sample((out_features,in_features))[:,:,0]
        self.b = norm_dist.sample((out_features,))[:,0]
        #print("self.w size is:",self.w.size())
        #print("self.b size is:", self.b.size())
        # ========================

        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def params(self):
        return [
            (self.w, self.dw), (self.b, self.db)
        ]

    def forward(self, x, **kw):
        """
        Computes an affine transform, y = x W^T + b.
        :param x: Input tensor of shape (N,Din) where N is the batch
        dimension, and Din is the number of input features, or of shape
        (N,d1,d2,...,dN) where Din = d1*d2*...*dN.
        :return: Affine transform of each sample in x.
        """

        x = x.reshape((x.shape[0], -1))

        # TODO: Compute the affine transform

        # ====== YOUR CODE: ======
        """
        Pavel done this:
        for each X_i tensor: out = X_i * w_T_i + b
        """
        w_T = torch.transpose(self.w, 0, 1)

        #print("x", x.size())
        #print("w_T", w_T.size())

        xW_T = torch.mm(x,w_T)
        #print("xW_T size",xW_T.size())
        #print("b size", self.b.size())
        b_expanded = self.b.expand_as(xW_T)
        #print("b_expanded size", b_expanded.size())
        out = xW_T+b_expanded
        # ========================

        self.grad_cache['x'] = x
        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to block output, shape (N, Dout).
        :return: Gradient with respect to block input, shape (N, Din)
        """
        x = self.grad_cache['x']

        # TODO: Compute
        #   - dx, the gradient of the loss with respect to x
        #   - dw, the gradient of the loss with respect to w
        #   - db, the gradient of the loss with respect to b
        # You should accumulate gradients in dw and db.
        # ====== YOUR CODE: ======
        """
        Pavel done this:
        for each X_i tensor: out = X_i * w_T_i + b
        
        x ----> | f() = xw+b | ----> out
        
        dx = dout * dout/dx  <---- | f(x) = xw+b | <---- dout
        dw = dout * dout/dw  <---- | f(x) = xw+b | <---- dout
        db = dout * dout/db  <---- | f(x) = xw+b | <---- dout
        
        thus: according to rule
        d(aT x)/dx = d(xT a) = a
               
        dout/dx = d(xwT+b)/dx = wT 
        dout/dw = d(xwT+b)/dw = x
        dout/db = d(xwT+b)/db = 1
        
        thus:
         
        dx = dout * w
        dw = dout * x
        db = dout * 1
        """

        temp_dw = torch.mm(torch.transpose(dout,0,1),x)
        temp_db = dout
        dx = torch.mm(dout,self.w)

        #print("self.dw size vs temp_dw size:",self.dw.size(),temp_dw.size())
        #print("self.db size vs temp_db size:", self.db.size(), temp_db.size())
        #print("dx size - should be (N, Din):",dx.size())

        self.dw = self.dw + temp_dw

        #   accumulate all rows of temp_db
        for row in range(temp_db.size(0)):
            self.db = self.db + temp_db[row]

        # ========================

        return dx

    def __repr__(self):
        return f'Linear({self.in_features}, {self.out_features})'


class ReLU(Block):
    """
    Rectified linear unit.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, **kw):
        """
        Computes max(0, x).
        :param x: Input tensor of shape (N,*) where N is the batch
        dimension, and * is any number of other dimensions.
        :return: ReLU of each sample in x.
        """

        # TODO: Implement the ReLU operation.
        # ====== YOUR CODE: ======
        """
        Pavel done this:
        just zero all negatives
        """
        zeros = torch.zeros_like(x)
        out = torch.max(x, zeros)

        # this doesn't work because some gradient issues:
        #out[out < 0] = 0

        # ========================

        self.grad_cache['x'] = x
        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to block output, shape (N, *).
        :return: Gradient with respect to block input, shape (N, *)
        """
        x = self.grad_cache['x']

        # TODO: Implement gradient w.r.t. the input x
        # ====== YOUR CODE: ======
        """
        Pavel done this:
        
        x ------> | max(0,x) | ------> out
        
        thus:
        
        dx = dout * dout/dx  <------ | max(0,x) | <------ dout
        
        while:
        dout/dx = if(out<0)=0 , if(out=>0)=1 
        
        NOTICE: ReLU is element wise
        """
        zeros = torch.zeros_like(dout)
        dx = torch.max(dout, zeros)
        # ========================

        return dx

    def params(self):
        return []

    def __repr__(self):
        return 'ReLU'


class CrossEntropyLoss(Block):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """
        Computes cross-entropy loss directly from class scores.
        Given class scores x, and a 1-hot encoding of the correct class yh,
        the cross entropy loss is defined as: -yh^T * log(softmax(x)).

        This implementation works directly with class scores (x) and labels
        (y), not softmax outputs or 1-hot encodings.

        :param x: Tensor of shape (N,D) where N is the batch
        dimension, and D is the number of features. Should contain class
        scores, NOT PROBABILITIES.
        :param y: Tensor of shape (N,) containing the ground truth label of
        each sample.
        :return: Cross entropy loss, as if we computed the softmax of the
        scores, encoded y as 1-hot and calculated cross-entropy by
        definition above. A scalar.
        """

        N = x.shape[0]
        xmax, _ = torch.max(x, dim=1, keepdim=True)
        x = x - xmax  # for numerical stability

        # TODO: Compute the cross entropy loss using the last formula from the
        # notebook (i.e. directly using the class scores).
        # Tip: to get a different column from each row of a matrix tensor m,
        # you can index it with m[range(num_rows), list_of_cols].
        # ====== YOUR CODE: ======
        """ 
        pavel done this:
        
        X - class scores
        loss = -Xy + log( sum on k(classes) of: exp( Xk ) )
        
        we have: 
        x: Tensor of shape (N,D) where N is the batch
        dimension, and D is the number of features.
        
        y: Tensor of shape (N,) containing the ground truth
        
        we want to get:
        loss (scalar) of a batch?
        """

        loss = 0

        for sample_idx in range(N):
            # TODO: calc loss
            curr_sample = x[sample_idx, :]

            # calc Xy
            #print("==========================")
            #print("sample idx: = ", sample_idx)
            #print("current sample: =",curr_sample)
            curr_label = y[sample_idx]
            #print("curr_label: = ", curr_label)
            Xy = curr_sample[curr_label]
            #print("Xy = ",Xy)

            # calc log sum
            exp_vec = torch.exp(curr_sample)
            log_sum = torch.log(torch.sum(exp_vec))
            #print("log_sum = ",log_sum)

            # calc loss - this is not good. We don't need sum.
            loss = loss + (-Xy.float() + log_sum)
        loss = loss/N


        # ========================

        self.grad_cache['x'] = x
        self.grad_cache['y'] = y
        return loss

    def backward(self, dout=1.0):
        """
        :param dout: Gradient with respect to block output, a scalar which
        defaults to 1 since the output of forward is scalar.
        :return: Gradient with respect to block input (only x), shape (N,D)
        """
        x = self.grad_cache['x']
        y = self.grad_cache['y']
        N = x.shape[0]

        # TODO: Calculate the gradient w.r.t. the input x
        # ====== YOUR CODE: ======
        """ pavel done this:
        
        we have this:
        
        x -----> | f = loss_func | -----> out
        
        dx = dout * dout/dx  <----- | f = loss_func | <----- dout (scalar)
                
        when:
        
        dout/dx = df/dx
        
        let's calc derivative of loss func:
        we have: -Xy + log(sum Xk)
        
        for dXi when i=y:
        dout/dXi = -1 + exp(Xy)/sum_k_on(exp(Xk))
        
        for dXi when i!=y:
        dout/dXi = exp(Xi)/sum_k_on(exp(Xk))
        """

        #print("x size",x.size())
        #print("y size", y.size())

        dx = torch.ones_like(x)
        dx = torch.mul(dx, dout)

        for sample_idx in range(N):
            # TODO: calc dx
            curr_sample = x[sample_idx, :]

            #print("==========================")
            #print("sample idx: = ", sample_idx)
            #print("current sample: =",curr_sample)
            curr_label = y[sample_idx]
            #print("curr_label: = ", curr_label)

            # calc log sum
            exp_vec = torch.exp(curr_sample)
            sum_k = torch.sum(exp_vec)
            #print("log_sum = ",log_sum)


            for i in range(x.size(1)):
                #print("i=",i)
                if curr_label == i:
                    temp_dout_dx = -1/N + torch.exp(curr_sample[i]) / (N*sum_k)
                else:
                    temp_dout_dx = torch.exp(curr_sample[i]) / (N*sum_k)
                dx[sample_idx,i] = dx[sample_idx,i] * temp_dout_dx
            #print("dx[sample_idx,:]",dx[sample_idx,:])
        #print("dx",dx)
        # ========================

        return dx

    def params(self):
        return []


class Dropout(Block):
    def __init__(self, p=0.5):
        super().__init__()
        assert 0. <= p <= 1.
        self.p = p

    def forward(self, x, **kw):
        # TODO: Implement the dropout forward pass. Notice that contrary to
        # previous blocks, this block behaves differently a according to the
        # current mode (train/test).
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

        return out

    def backward(self, dout):
        # TODO: Implement the dropout backward pass.
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

        return dx

    def params(self):
        return []

    def __repr__(self):
        return f'Dropout(p={self.p})'


class Sequential(Block):
    """
    A Block that passes input through a sequence of other blocks.
    """
    def __init__(self, *blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x, **kw):
        out = None

        # TODO: Implement the forward pass by passing each block's output
        # as the input of the next.
        # ====== YOUR CODE: ======
        temp_input = x
        for block in self.blocks:
            temp_input = block.forward(temp_input)
        out = temp_input
        # ========================

        return out

    def backward(self, dout):
        din = None

        # TODO: Implement the backward pass.
        # Each block's input gradient should be the previous block's output
        # gradient. Behold the backpropagation algorithm in action!
        # ====== YOUR CODE: ======
        temp_dout = dout
        #print("num of blocks = ",len(self.blocks))
        for block_idx in range(len(self.blocks)-1,-1,-1):
            #print("temp_dout",temp_dout)
            #print("block_idx",block_idx)
            temp_dout = self.blocks[block_idx].backward(temp_dout)
        din = temp_dout
        # ========================

        return din

    def params(self):
        params = []

        # TODO: Return the parameter tuples from all blocks.
        # ====== YOUR CODE: ======
        for block in self.blocks:
            #print("block.params() type",type(block.params()))
            #print("block.params() length",len(block.params()))
            if len(block.params()) > 1:
                for tup in block.params():
                    params.append(tup)
            #else:
                #params.append((None,None))
        # ========================

        return params

    def train(self, training_mode=True):
        for block in self.blocks:
            block.train(training_mode)

    def __repr__(self):
        res = 'Sequential\n'
        for i, block in enumerate(self.blocks):
            res += f'\t[{i}] {block}\n'
        return res

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, item):
        return self.blocks[item]

