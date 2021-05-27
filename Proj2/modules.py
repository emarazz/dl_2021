"""
Modules for Project 2 - Mini deep-learning framework.
Deep Learning EE-559, Fall 2021.
"""

import torch
from torch import Tensor

# Set autograd off
torch.set_grad_enabled(False)

class Tanh:
    """ Creates a module for the Tanh function and its respective derivative. """

    def __call__(self, x):
        """ Applies the tanh function to the input. """

        return x.tanh()
    
    def derivative(self, x):
        """ Applies the derivative of the tanh function to the input. """

        return 1 - x.tanh().pow(2)

class ReLU:
    """ Creates a module for the ReLU function and its respective derivative. """
    
    def __call__(self, x):
        """ Applies the ReLU function to the input. """

        idx = x < 0
        x_copy = x.clone() # Avoid changing x
        x_copy[idx] = 0
        return x_copy
    
    def derivative(self, x):
        """ Applies the derivative of the ReLU function to the input. """

        idx = x < 0
        x_copy = x.clone() # Avoid changing x
        x_copy[idx] = 0
        x_copy[torch.logical_not(idx)] = 1
        return x_copy


class MSELoss():
    """ Module for the MSE loss. """

    def __init__(self, reduction='mean'):
        """ Initializes the MSE with the chosen reduction: 'mean' or 'sum'. """

        self.reduction = reduction

    def __call__(self, prediction, target):
        """ Returns the MSE with the chosen reduction.
        The mean or sum are applied to all the elements of the error, where: error = prediction - target.
            Parameters:
                prediction : the output of the forward pass.
                target : the target value.
        """

        # print(target.shape)
        # print(prediction.shape)
        error = target - prediction

        if self.reduction == 'mean':
            return error.pow(2).mean()

        elif self.reduction == 'sum':
            return error.pow(2).sum()
    
    def grad(self, prediction, target):
        """ Returns the gradient of the MSE w.r.t. the prediction. """

        error = target - prediction

        if self.reduction == 'mean':
            # return 2 / (error.size(0) * error.size(1)) * error
            return - 2 / torch.flatten(error).size(0) * error

        elif self.reduction == 'sum':
            return - 2 * error.view(error.size(0),-1)


class Linear():
    """ Module for an affine transformation: output = input @ weight.T + bias.
    The gradients of the loss w.r.t. the respecetives weight and bias are stored during the backpropagation.
    """

    def __init__(self, in_features, out_features):
        """ Creates the weight and bias for the affine transformation and their gradients.
        Attributes weight and bias are initialized similarly to PyTorch's default initialization.
        Attributes weight_grad and bias_grad initialized with zeros.
            Parameters:
                in_features : number of features of the input.
                out_features : number of features of the output.
        """

        self.input = None
        self.output = None
        self.weight = torch.empty(out_features, in_features).uniform_(  - torch.tensor(1 / in_features).sqrt(),
                                                                        torch.tensor(1 / in_features).sqrt()    )
        self.bias = torch.empty(out_features).uniform_( - torch.tensor(1 / in_features).sqrt(),
                                                        torch.tensor(1 / in_features).sqrt()    )
        self.weight_grad = torch.zeros(out_features, in_features)
        self.bias_grad = torch.zeros(out_features, in_features)
        
    def __call__(self, input):
        """ Applies the affine transformation to the input.
        Both, input and output are stored in the module.
        """
        
        self.input = input
        self.output = self.input @ self.weight.T + self.bias
        return self.output


class Sequential():
    """ Sequential model for a MLP (Multi Layer Perceptron). """

    def __init__(self, *args):
        """ Initializes the object.
            Parameters:
                args : modules sorted following order: linear, act_fun, ..., linear, act_fun, loss_fun.
                       e.g. Linear(), Tanh(), ... ,Linear(), ReLU(), MSELoss().
        """

        self.training = True
        self.args = args
        self.prediction = None
        self.target = None
        

    def train(self):
        """ Training mode for the model.
        During the loss computation, it will store the prediction and target values in the model.
        Those values are used in the backward step.
        """

        self.training = True
        
    def eval(self):
        """ Evaluation mode for the model.
        During the loss computation, it will set the prediction and target values to None in the model,
        Those values are used in the the backward step and therefore backpropagation will not be possible.
        """

        self.training = False

    def forward(self, input):
        """ Forward step.
        Applies all the operations declared in the model to the input, but the loss function.
        """

        for arg in self.args[:-1]:
            input = arg(input)
        return input

    def loss(self, prediction ,target):
        """ Returns the loss.
        Both, the prediction and target are stored in the model.
        """

        if self.training:
            self.prediction = prediction
            self.target = target
        else:
            self.prediction = None
            self.target = None

        return self.args[-1](prediction, target)

    def backward(self):
        """ Backward step.
        The gradients are calculated and stored in their respectives modules.
        self.training must be True.
        """
        assert(self.training)

        # Use easier names for the variables. (Optional improvement)
        # Calculates de gradients for the last layer.
        dl_dxL = self.args[-1].grad(self.prediction, self.target)
        
        dl_dsl = dl_dxL * self.args[-2].derivative(self.args[-3].output)
        self.args[-3].weight_grad = dl_dsl.T @ self.args[-3].input 
        self.args[-3].bias_grad = dl_dsl.T @ torch.ones( self.target.size(0) )

        # Backwards loop. Use easier names for the variables (Optional improvement)
        # The loop goes from self.args[-3] to self.args[1] in steps of -2. 
        # Calculates the gradients for the remaining gradients.
        for idx in range(len(self.args[:-3]), 1, -2):
            dl_dsl = (dl_dsl @ self.args[idx].weight) * self.args[idx-1].derivative(self.args[idx-2].output)
            self.args[idx-2].weight_grad = dl_dsl.T @ self.args[idx-2].input 
            self.args[idx-2].bias_grad = dl_dsl.T @ torch.ones( self.target.size(0) )

        return

    def __getitem__(self, items):
        """ Magic method to use brackets [] to get items. """

        return self.args[items]
    
    def with_parameters(self):
        """ Returns the modules of the model that contain the weights and bias. """

        return  self.args[0:-2:2]
    
    def zero_grad(self):
        """ Set the weights and bias of the model to zero. """

        for arg in self.with_parameters():
            arg.weight_grad.zero_()
            arg.bias_grad.zero_()
        return 
