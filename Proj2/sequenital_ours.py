import torch
from torch import Tensor

import dlc_practical_prologue as prologue


class Tanh:
    '''Creates a submodule for the Tanh function and its respective derivative.'''

    def __call__(self, x):
        '''Applies the tanh function to the input.'''

        return x.tanh()
    
    def derivative(self, x):
        '''Applies the derivative of the tanh function to the input.'''

        return 1 - x.tanh().pow(2)

class ReLU:
    '''Creates a submodule for the ReLU function and its respective derivative.'''
    
    def __call__(self, x):
        '''Applies the ReLU function to the input.'''

        idx = x < 0
        x_copy = x.clone() # Avoid changing x
        x_copy[idx] = 0
        return x_copy
    
    def derivative(self, x):
        '''Applies the derivative of the ReLU function to the input.'''

        idx = x < 0
        x_copy = x.clone() # Avoid changing x
        x_copy[idx] = 0
        x_copy[~idx] = 1
        return x_copy


class MSELoss():
    '''Submodule for the MSE. Note that the loss can be obtained'''

    def __init__(self, reduction='mean'):
        '''Initializes the MSE with the chosen reduction: mean or sum. 
        The mean or sum are applied to all the elements of the error calculated. 
        '''
        self.reduction = reduction

    def __call__(self, predictions, target):
        '''Returns the MSE with the chosen reduction.'''

        if self.reduction == 'mean':
            error = target - predictions
            return error.pow(2).mean()

        elif self.reduction == 'sum':
            error = target - predictions
            return error.pow(2).sum()
    
    def grad(self, prediction, target):
        '''Returns the gradient of the MSE w.r.t. the prediction.'''

        if self.reduction == 'mean':
            error = target - prediction
            return 2 /(error.size(0) * error.size(1)) * error

        elif self.reduction == 'sum':
            error = target - prediction
            return 2 * error.view(error.size(0),-1)


class Linear():
    '''Submodule for an affine transformation: output = input @ weight.T + bias.
    The gradients of the loss w.r.t. the respecetives weight and bias are stored during the backpropagation.
    '''

    def __init__(self, in_features, out_features):
        '''Creates the weight and bias for the affine transformation. Both are initialized with Xavier Initialization.
        Creates weight_grad and bias_grad initialized with zeros.
        '''

        self.input = None
        self.output = None
        self.weight = torch.empty(out_features, in_features).uniform_(
                                    - torch.tensor(1 / in_features).sqrt(),
                                    torch.tensor(1 / in_features).sqrt())
        self.bias = torch.empty(out_features).uniform_(
                                    - torch.tensor(1 / in_features).sqrt(),
                                    torch.tensor(1 / in_features).sqrt())
        self.weight_grad = torch.zeros(out_features, in_features)
        self.bias_grad = torch.zeros(out_features, in_features)
        
    def __call__(self, input):
        '''Applies the affine transformation to the input: output = input @ weight.T + bias and stores the input and output.'''
        
        self.input = input
        self.output = input @ self.weight.T + self.bias
        return self.output


class Sequential():
    '''Creates a sequential model for a Multi Layer Perceptron.
    The args must be sorted in the following order: Linear(), Tanh(), ... ,Linear(), ReLU(), MSELoss().
    '''

    def __init__(self, *args):
        '''Initializes the object. Notice that self.prediction and self.target are initialized with None.'''
               
        self.args = args
        self.prediction = None
        self.target = None

    def forward(self, input):
        '''Forward step. Applies all the operations declared in the object to x.'''

        for arg in self.args[:-1]:
            input = arg(input)
        return input

    def loss(self, prediction ,target):
        '''Returns the loss and assigns values to self.prediction and self.target.'''
        self.prediction = prediction
        self.target = target
        return self.args[-1](prediction, target)

    def backward(self):
        '''Backward step. The gradients are assigned to their respective parameters.'''

        # Use easier names for the variables.
        # Calculates de gradients for the last layer.
        dl_dxL = - self.args[-1].grad(self.prediction, self.target)
        
        dl_dsl = dl_dxL * self.args[-2].derivative(self.args[-3].output)
        self.args[-3].weight_grad = dl_dsl.T @ self.args[-3].input 
        self.args[-3].bias_grad = dl_dsl.T @ torch.ones( self.target.size(0) )

        # Backwards loop. Use easier names for the variables
        # Calculates the gradients for the remaining gradients.
        for idx in range(len(self.args[:-3]), 1, -2):
            dl_dsl = (dl_dsl @ self.args[idx].weight) * self.args[idx-1].derivative(self.args[idx-2].output)
            self.args[idx-2].weight_grad = dl_dsl.T @ self.args[idx-2].input 
            self.args[idx-2].bias_grad = dl_dsl.T @ torch.ones( self.target.size(0) )

        return

    def __getitem__(self, items):
        '''Magic method to use brackets [] to get items.'''

        return self.args[items]
    
    
    def parameters(self):
        '''Returns the objects that contain the weights and bias.'''
        return  self.args[0:-2:2]
    
    def zero_grad(self):
        '''Set the weights and bias of the model to zero.'''

        for arg in self.parameters():
            arg.weight_grad.zero_()
            arg.bias_grad.zero_()
        return 


train_input, train_target, test_input, test_target = prologue.load_data(one_hot_labels = True,
                                                                        normalize = True)
nb_classes = train_target.size(1)
nb_train_samples = train_input.size(0)

zeta = 0.90

train_target = train_target * zeta
test_target = test_target * zeta


torch.manual_seed(0)


model = Sequential(
    Linear(784, 100),Tanh(),
    Linear(100,100), Tanh(),
    Linear(100,100), Tanh(),
    Linear(100,10), ReLU(),
    MSELoss(reduction='sum')
)

nb_epochs = 1000
batch_size = 1000
nb_hidden = 50
eta = 1e-1 / nb_train_samples
epsilon = 1e-6

for e in range(nb_epochs):

    #### test predictions should be placed CAREFULLY because it stores values!!!
    test_prediction = model.forward(test_input) 
    test_loss = model.loss(test_prediction, test_target)
    ####

    prediction = model.forward(train_input) # inputs and outputs of linear layers are stored in Lines() objects
    loss = model.loss(prediction, train_target) # prediction and train_target are stored in Sequential() object  
    print('epoch: {:4d} train_loss: {:8.4f} test_loss {:8.4f}'.format(e+1, loss, test_loss)) 

    model.zero_grad() # gradeints of linear layers are set to zero
    model.backward()# backpropagation - gradients are calculated and stored based on the prediction and target set in model.loss().
                    # backpropagation - the calculated gradients are stored as attributes in the object. 

    for p in model.parameters():
        p.weight = p.weight - eta * p.weight_grad
        p.bias = p.bias - eta * p.bias_grad
            
