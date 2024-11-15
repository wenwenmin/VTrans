import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.autograd import Variable
import datetime
import random
import time
import logging
LOGGER = logging.getLogger()
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import os
from TransVCOX.pre_train.vae_main import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cancer_type_list = ['brca', 'cesc', 'chol', 'coad', 'dlbc', 'esca', 'gbm', 'hnsc', 'kich', 'kirc', 'kirp', 'lgg', 'lihc', 'luad', 'lusc', 'meso', 'ov', 'paad', 'pcpg', 'prad', 'sarc', 'skcm', 'stad', 'tgct', 'thca', 'thym', 'ucec', 'ucs', 'uvm']


scaler = StandardScaler()
#scaler = MinMaxScaler()


class reparametrize(nn.Module):
    def __init__(self):
        super(reparametrize, self).__init__()

    def forward(self, z_mean, z_log_var):
        epsilon = torch.randn(z_mean.shape)
        epsilon = epsilon.to(device)
        return z_mean + (z_log_var / 2).exp() * epsilon

#256, 100, 0.5
class VaeEncoder(nn.Module):
    def __init__(self, config):
        super(VaeEncoder, self).__init__()
        self.Dense = nn.Linear(config.input_size, config.hidden_size_1)
        self.z_mean = nn.Linear(config.hidden_size_1, config.hidden_size_2)
        self.z_log_var = nn.Linear(config.hidden_size_1, config.hidden_size_2)
        self.dropout = nn.Dropout(p=config.vae_dropout_rate)
        self.sample = reparametrize()

    def forward(self, x):
        o = torch.nn.functional.relu(self.Dense(x))
        o = self.dropout(o)
        z_mean = self.z_mean(o)
        z_log_var = self.z_log_var(o)
        o = self.sample(z_mean, z_log_var)
        return o, z_mean, z_log_var


class VaeDecoder(nn.Module):
    def __init__(self, config):
        super(VaeDecoder, self).__init__()
        self.Dense = nn.Linear(config.hidden_size_2, config.hidden_size_1)
        self.out = nn.Linear(config.hidden_size_1, config.input_size)
        self.dropout = nn.Dropout(p=config.vae_dropout_rate)
        #self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, z):
        o = nn.functional.relu(self.Dense(z))
        o = self.dropout(o)
        o = self.out(o)
        return self.relu(o)


class Vae(nn.Module):
    def __init__(self, config):
        super(Vae, self).__init__()
        self.best_val_loss = 999.999
        self.encoder = VaeEncoder(config)
        self.decoder = VaeDecoder(config)

    def forward(self, x):
        o, mean, var = self.encoder(x)
        return self.decoder(o), mean, var


model = Vae(config=config)
model = model.to(device)
print(model)