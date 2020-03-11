import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
import torch.utils.data as data


class Net(torch.nn.Module):

    def __init__(self,
                N,
                n_in,
                n_out,
                layer_sizes=[256,],
                dropout_rate=0.5,
                tau=0.1,
                ):

        super(Net, self).__init__()

        # Dimensions of the problem
        self.n_in  = n_in
        self.n_out = n_out
        self.n_layers=len(layer_sizes)

        # Params
        self.dropout_rate = dropout_rate
        self.tau = tau
        lengthscale = 1e-2
        self.reg = lengthscale**2 * (1 - dropout_rate) / (2. * N * tau)

        # Layers
        self.model=torch.nn.Sequential()

        self.model.add_module("Dropout0",torch.nn.Dropout(p=self.dropout_rate, inplace=False))
        self.model.add_module("Linear0",torch.nn.Linear(self.n_in,layer_sizes[0],bias=True))
        self.model.add_module("Relu",torch.nn.ReLU())

        if len(layer_sizes)>0:
            for i in range(len(layer_sizes)-1):
                self.model.add_module("Dropout{}".format(i+1),torch.nn.Dropout(p=self.dropout_rate, inplace=False))
                self.model.add_module("Linear{}".format(i+1),torch.nn.Linear(layer_sizes[i],layer_sizes[i+1],bias=True))
                self.model.add_module("Relu",torch.nn.ReLU())

        self.model.add_module("Dropout{}".format(len(layer_sizes)),torch.nn.Dropout(p=self.dropout_rate, inplace=False))
        self.model.add_module("Linear{}".format(len(layer_sizes)),torch.nn.Linear(layer_sizes[-1],n_out,bias=True))



    def forward(self,x):
        """
        forward pass
        """

        out=self.model(x)

        if self.n_out>1:
            out=F.log_softmax(out)

        return out
