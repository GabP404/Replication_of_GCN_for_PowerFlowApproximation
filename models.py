import math
from typing import Any
import dgl
import dgl.nn.pytorch.conv as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch

import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.modules.lazy import LazyModuleMixin

class HyperparameterGiver:
    def __init__(self):
        self.learning_rate = None
        self.hidden_size = None

class GcnHG(HyperparameterGiver):
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.001
        self.filter_map_convolutional_1 = 30
        self.filter_map_convolutional_2 = 30
        self.filter_map_convolutional_3 = 30
        self.filter_map_convolutional_4 = 20
        self.num_neighbors = 20

class Fcn1HG(HyperparameterGiver):
    def __init__(self, out_size):
        super().__init__()
        self.learning_rate = 0.001
        self.hidden_size = 10 * out_size
        self.dropout_rate = 0.5

class Fcn2HG(HyperparameterGiver):
    def __init__(self, out_size):
        super().__init__()
        self.learning_rate = 0.01
        self.hidden_size = out_size

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.metrics = None


class FCN1(CustomModel):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.metrics = Fcn1HG(out_size)
        # two-layer hidden FCN
        hid_size = self.metrics.hidden_size
        self.fc1 = nn.Linear(in_size, hid_size)
        self.fc2 = nn.Linear(hid_size, hid_size)
        self.fc3 = nn.Linear(hid_size, out_size)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(p=self.metrics.dropout_rate)

    def forward(self, features):
        x = self.fc1(features)
        x = self.relu(x)
        #x = self.dropout(x)  # Apply dropout after the activation
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.metrics.learning_rate)
        return optimizer


class FCN2(CustomModel):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.metrics = Fcn2HG(out_size)
        # two-layer FCN
        hid_size = self.metrics.hidden_size
        self.fc1 = nn.Linear(in_size, hid_size)
        self.fc2 = nn.Linear(hid_size, out_size)
        self.relu = nn.ReLU()

    def forward(self, features):
        x = self.fc1(features)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.metrics.learning_rate)
        return optimizer
    

class LocallyConnectedLayer(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, neighborhood=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        filter = []
        for row in neighborhood :
            temp = []
            for e in row :
                temp.append(int(e))
            filter.append(temp)
        self.myFilter = nn.Parameter(torch.tensor(filter), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        # self.weight.data = self.weight * self.myFilter

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight * self.myFilter, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    
    
class GCN(CustomModel):
    def __init__(self, in_features, out_features, neighborhood, neighborhood_out):
        super().__init__()

        self.metrics = GcnHG()
        src = []
        dst = []
        for i,row in enumerate(neighborhood_out):
            for j,e in enumerate(row):
                if i != j :
                    if e :
                        src.append(i)
                        dst.append(j)
        self.graph = dgl.graph((torch.tensor(src),torch.tensor(dst)))
        self.graph = dgl.add_self_loop(self.graph)
        self.lc1 = LocallyConnectedLayer(in_features, out_features, neighborhood=neighborhood)
        self.conv1 = dglnn.GraphConv(in_feats=out_features, out_feats=self.metrics.filter_map_convolutional_1, norm='both', weight=True, bias=True, activation=nn.ReLU())
        self.conv2 = dglnn.GraphConv(in_feats=self.metrics.filter_map_convolutional_1, out_feats=self.metrics.filter_map_convolutional_2, norm='both', weight=True, bias=True, activation=nn.ReLU())
        self.conv3 = dglnn.GraphConv(in_feats=self.metrics.filter_map_convolutional_2, out_feats=self.metrics.filter_map_convolutional_3, norm='both', weight=True, bias=True, activation=nn.ReLU())
        self.conv4 = dglnn.GraphConv(in_feats=self.metrics.filter_map_convolutional_3, out_feats=self.metrics.filter_map_convolutional_4, norm='both', weight=True, bias=True, activation=nn.ReLU())
        self.lc2 = LocallyConnectedLayer(out_features, out_features, neighborhood=neighborhood_out)
        self.relu = nn.ReLU()

    def forward(self, features):
        x = self.lc1(features)
        x = self.relu(x)
        out_conv1 = torch.empty(0, )
        for input in x :
            temp = self.conv1(self.graph, input.unsqueeze(0))
            out_conv1 = torch.cat((out_conv1, temp.unsqueeze(0)), 0)
        out_conv2 = torch.empty(0, )
        for input in out_conv1 :
            temp = self.conv2(self.graph, input)
            out_conv2 = torch.cat((out_conv2, temp.unsqueeze(0)), 0)
        out_conv3 = torch.empty(0, )
        for input in out_conv2 :
            temp = self.conv3(self.graph, input)
            out_conv3 = torch.cat((out_conv3, temp.unsqueeze(0)), 0)
        out_conv4 = torch.empty(0, )
        for input in out_conv3 :
            temp = self.conv4(self.graph, input)
            out_conv4 = torch.cat((out_conv4, temp.unsqueeze(0)), 0)
        out = torch.sum(out_conv4, -1)
        out = self.lc2(out)
        return out
    
    def get_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.metrics.learning_rate)