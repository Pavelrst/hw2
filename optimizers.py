import abc
import torch
from torch import Tensor


class Optimizer(abc.ABC):
    """
    Base class for optimizers.
    """
    def __init__(self, params):
        """
        :param params: A sequence of model parameters to optimize. Can be a
        list of (param,grad) tuples as returned by the Blocks, or a list of
        pytorch tensors in which case the grad will be taken from them.
        """
        assert isinstance(params, list) or isinstance(params, tuple)
        self._params = params

    @property
    def params(self):
        """
        :return: A sequence of parameter tuples, each tuple containing
        (param_data, param_grad). The data should be updated in-place
        according to the grad.
        """
        returned_params = []
        for x in self._params:
            if isinstance(x, Tensor):
                p = x.data
                dp = x.grad.data if x.grad is not None else None
                returned_params.append((p, dp))
            elif isinstance(x, tuple) and len(x) == 2:
                returned_params.append(x)
            else:
                raise TypeError(f"Unexpected parameter type for parameter {x}")

        return returned_params

    def zero_grad(self):
        """
        Sets the gradient of the optimized parameters to zero (in place).
        """
        for p, dp in self.params:
            dp.zero_()

    @abc.abstractmethod
    def step(self):
        """
        Updates all the registered parameter values based on their gradients.
        """
        raise NotImplementedError()


class VanillaSGD(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg

    def step(self):
        for p, dp in self.params:
            if dp is None:
                continue

            # TODO: Implement the optimizer step.
            # Update the gradient according to regularization and then
            # update the parameters tensor.
            # ====== YOUR CODE: ======
            """
            pavel done this:
            Update the gradient according to regularization
            Grad <-- Grad + 2 * lambda * W
            
            W <--- W - step * Grad
            """

            # TODO: there is a problem with updating the params tensor
            # TODO: we need to update the input parameter of a class.
            # TODO: see the test in Jupyter notebook to understand the issue.
            # TODO: how it's done in python?

            # TODO: "params" is an array - If you pass in an array and change it's
            # TODO: contents that will be reflected wherever else it is referenced

            # I think this is the grad update and step,
            # but I can't check it, because the param array
            # can't be updated. How to do this?!
            grad = dp.sub(2*self.reg) # not sure is sub or add?!
            p = p - torch.mul(grad,self.learn_rate)

            # ========================


class MomentumSGD(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0, momentum=0.9):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        :param momentum: Momentum factor
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg
        self.momentum = momentum

        # TODO: Add your own initializations as needed.
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

    def step(self):
        for p, dp in self.params:
            if dp is None:
                continue

            # TODO: Implement the optimizer step.
            # update the parameters tensor based on the velocity. Don't forget
            # to include the regularization term.
            # ====== YOUR CODE: ======
            raise NotImplementedError()
            # ========================


class RMSProp(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0, decay=0.99, eps=1e-8):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        :param decay: Gradient exponential decay factor
        :param eps: Constant to add to gradient sum for numerical stability
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg
        self.decay = decay
        self.eps = eps

        # TODO: Add your own initializations as needed.
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

    def step(self):
        for p, dp in self.params:
            if dp is None:
                continue

            # TODO: Implement the optimizer step.
            # Create a per-parameter learning rate based on a decaying moving
            # average of it's previous gradients. Use it to update the
            # parameters tensor.
            # ====== YOUR CODE: ======
            raise NotImplementedError()
            # ========================
