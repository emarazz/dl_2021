import torch
from torch import nn
from torch.nn import functional as F  

"""
Models 
    - BaseNetCNN
    - BaseNetMLP
    - AuxNet
    - SiameseNet
"""
class BaseNetMLP(nn.Module):
    def __init__(self, nb_hidden1 = 512, nb_hidden2 = 512, nb_hidden = 'NaN', dropout_prob=0):
        """
        Baseline A: MultiLayer Perceptron 
                    No auxualiary losses
                    No weight sharing
            nb_hidden1: number of hidden units first fully connected layer | default = 512
            nb_hidden2: number of hidden units second fully connected layer | default = 512
            nb_hidden3: unused input | default = 'NaN' 
            dropout_prob: dropout probability | default = 0
        """
        super().__init__()

        self.nb_hidden1 = nb_hidden1
        self.nb_hidden2 = nb_hidden2
        self.dropout_prob = dropout_prob

        self.fc1 = nn.Linear(2*14*14, self.nb_hidden1)
        self.batchNorm1 = nn.BatchNorm1d(self.nb_hidden1)
        self.fc2 = nn.Linear(self.nb_hidden1, self.nb_hidden2)
        self.batchNorm2 = nn.BatchNorm1d(self.nb_hidden2)
        self.fc3 = nn.Linear(self.nb_hidden2, 2)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x.view(-1, 2*14*14))
        x = self.relu(self.batchNorm1(x))
        x = self.dropout(x)
        x = self.relu(self.batchNorm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
    
    def get_str_parameters(self):
        return '{} - nb_hidden1={}, nb_hidden2={}, dropout={}'.format(
                type(self).__name__, self.nb_hidden1, self.nb_hidden2, self.dropout_prob)

    def get_aux_info(self):
        return False

class BaseNetCNN(nn.Module):
    def __init__(self, nb_hidden1 = 64, nb_hidden2 = 64, nb_hidden3 = 'NaN', dropout_prob=0):
        """
        Baseline B: Convolutional Neural Network + Feed Forward Netork 
                    No auxualiary losses
                    No weight sharing
            nb_hidden1: number of hidden units first fully connected layer | default = 64
            nb_hidden2: number of hidden units second fully connected layer | default = 64
            nb_hidden3: unused input | default = 'NaN' 
            dropout_prob: dropout probability | default = 0
        """
        super().__init__()

        self.nb_hidden1 = nb_hidden1
        self.nb_hidden2 = nb_hidden2
        self.dropout_prob = dropout_prob
        
        # feature extractor
        self.conv1 = nn.Conv2d( 2, 32, kernel_size=5) 
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5) 
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3) 
        self.batchNorm3 = nn.BatchNorm2d(64)
        
        # 0-1 classifier 
        self.fc1 = nn.Linear(256, self.nb_hidden1)
        self.batchNorm4 = nn.BatchNorm1d(self.nb_hidden1)
        self.fc2 = nn.Linear(self.nb_hidden1, self.nb_hidden2)
        self.batchNorm5 = nn.BatchNorm1d(self.nb_hidden2)
        self.fc3 = nn.Linear(self.nb_hidden2, 2)

        self.max_pool = nn.MaxPool2d(kernel_size = 2)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, x): 
        x = self.relu(self.batchNorm1(self.conv1(x)))
        x = self.relu(self.batchNorm2(self.conv2(x)))
        x = self.relu(self.batchNorm3(self.conv3(x)))
        x = self.max_pool(x)                                    
        x = self.fc1(x.view(-1, 256))                           
        x = self.relu(self.batchNorm4(x))
        x = self.dropout(x) 
        x = self.relu(self.batchNorm5(self.fc2(x)) ) 
        x = self.dropout(x)  
        x = self.fc3(x) 
        return x
    
    def get_str_parameters(self):
        return '{} - nb_hidden1={}, nb_hidden2={}, dropout={}'.format(
                type(self).__name__, self.nb_hidden1, self.nb_hidden2, self.dropout_prob)

    def get_aux_info(self):
        return False

class AuxNet(nn.Module):
    def __init__(self, nb_hidden1 = 64, nb_hidden2 = 256, nb_hidden3 = 64, dropout_prob=0):
        """
        AuxNet: Convolutional Neural Network + Feed Forward Netork 
                    Auxualiary losses
                    No weight sharing
            nb_hidden1: number of hidden units first fully connected layer | default = 64
            nb_hidden2: number of hidden units second fully connected layer | default = 256
            nb_hidden3: number of hidden units fourth fully connected layer | default = 64
            dropout_prob: dropout probability | default = 0
        """
        super().__init__()

        self.nb_hidden1 = nb_hidden1
        self.nb_hidden2 = nb_hidden2
        self.nb_hidden3 = nb_hidden3
        self.dropout_prob = dropout_prob

        self.feat_extractor1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),     
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3), 
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(64, 64, kernel_size=3), 
            nn.BatchNorm2d(64), 
            nn.ReLU()
        )
        self.feat_extractor2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),    
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3), 
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2), 
            nn.Conv2d(64, 64, kernel_size=3), 
            nn.BatchNorm2d(64), 
            nn.ReLU()
        )
        self.classifier1 = nn.Sequential( 
            nn.Dropout(dropout_prob),
            nn.Linear(256, nb_hidden1), 
            nn.BatchNorm1d(nb_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(nb_hidden1, nb_hidden2), 
            nn.BatchNorm1d(nb_hidden2),
            nn.ReLU(),
            nn.Linear(nb_hidden2,10), 
            nn.BatchNorm1d(10)
        )
        self.classifier2 = nn.Sequential( 
            nn.Dropout(dropout_prob),
            nn.Linear(256, nb_hidden1), 
            nn.BatchNorm1d(nb_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(nb_hidden1, nb_hidden2),
            nn.BatchNorm1d(nb_hidden2),
            nn.ReLU(),
            nn.Linear(nb_hidden2,10),
            nn.BatchNorm1d(10)
        )
        self.final_classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(20, nb_hidden3),
            nn.BatchNorm1d(nb_hidden3),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(nb_hidden3, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
    
    def forward(self, x):
        # input separation
        x1 = x[:, 0, :, :].view((x.shape[0], 1) + tuple(x.shape[2:]))
        x2 = x[:, 1, :, :].view((x.shape[0], 1) + tuple(x.shape[2:]))
        # feature extraction
        x1 = self.feat_extractor1(x1)
        x2 = self.feat_extractor2(x2)
        # 0-9 classifier 
        x1 = self.classifier1(x1.view(-1, 256))
        x2 = self.classifier2(x2.view(-1, 256))
        # 0-1 classifier
        y = torch.cat((x1, x2), 1)
        y = self.final_classifier(y)
        
        return x1, x2, y
    
    def get_str_parameters(self):
        return '{} - nb_hidden1={}, nb_hidden2={}, nb_hidden3={}, dropout={}'.format(
                type(self).__name__, self.nb_hidden1, self.nb_hidden2, self.nb_hidden3, self.dropout_prob)
    
    def get_aux_info(self):
        return True

class SiameseNet(nn.Module):
    def __init__(self, nb_hidden1 = 64, nb_hidden2 = 128, nb_hidden3 = 128, dropout_prob=0):
        """
        SiameseNet: Convolutional Neural Network + Feed Forward Netork 
                    Auxualiary losses
                    Weight sharing
            nb_hidden1: number of hidden units first fully connected layer | default = 64
            nb_hidden2: number of hidden units second fully connected layer | default = 128
            nb_hidden3: number of hidden units fourth fully connected layer | default = 128
            dropout_prob: dropout probability | default = 0
        """
        super().__init__()

        self.nb_hidden1 = nb_hidden1
        self.nb_hidden2 = nb_hidden2
        self.nb_hidden3 = nb_hidden3
        self.dropout_prob = dropout_prob        

        self.feat_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),   
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2), 
            nn.Conv2d(64, 64, kernel_size=3), 
            nn.BatchNorm2d(64), 
            nn.ReLU()
        )
        self.classifier = nn.Sequential( 
            nn.Dropout(dropout_prob),
            nn.Linear(256, nb_hidden1),
            nn.BatchNorm1d(nb_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(nb_hidden1, nb_hidden2),
            nn.BatchNorm1d(nb_hidden2),
            nn.ReLU(),
            nn.Linear(nb_hidden2,10),
            nn.BatchNorm1d(10)
        )
        self.final_classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(20, nb_hidden3), 
            nn.BatchNorm1d(nb_hidden3),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(nb_hidden3, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
    
    def forward(self, x):
        # input separation
        x1 = x[:, 0, :, :].view((x.shape[0], 1) + tuple(x.shape[2:]))   # digit A
        x2 = x[:, 1, :, :].view((x.shape[0], 1) + tuple(x.shape[2:]))   # digit B
        # features extraction
        x1 = self.feat_extractor(x1) 
        x2 = self.feat_extractor(x2)
        # 0-9 classifier
        x1 = self.classifier(x1.view(-1, 256))
        x2 = self.classifier(x2.view(-1, 256))
        # 0-1 classifier
        y = torch.cat((x1, x2), 1)
        y = self.final_classifier(y)
        
        return x1, x2, y

    def get_str_parameters(self):
        return '{} - nb_hidden1={}, nb_hidden2={}, nb_hidden3={}, dropout={}'.format(
                type(self).__name__, self.nb_hidden1, self.nb_hidden2, self.nb_hidden3, self.dropout_prob)

    def get_aux_info(self):
        return True



        