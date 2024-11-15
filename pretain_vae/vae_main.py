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
from TransVCOX.pre_train.vae_model import *
import argparse

parser = argparse.ArgumentParser()


# VAE
parser.add_argument('--input_size', '-is', type=int, default=1406)
parser.add_argument('--hidden_size_1', '-hs1', type=int, default=256)
parser.add_argument('--hidden_size_2', '-hs2', type=int, default=20)
parser.add_argument('--vae_dropout_rate', '-vdr', type=float, default=0.1)
parser.add_argument('--vae_learning_rate', '-vlr', type=float, default=1e-3)
parser.add_argument('--L1_regularization', '-l1', type=float, default=1e-2)
parser.add_argument('--scheduler_factor', '-sf', type=float, default=0.1)
parser.add_argument('--scheduler_patience', '-sp', type=int, default=10)
parser.add_argument('--number_epochs', '-ne', type=int, default=80)

config = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cancer_type_list = ['brca', 'cesc', 'chol', 'coad', 'dlbc', 'esca', 'gbm', 'hnsc', 'kich', 'kirc', 'kirp', 'lgg', 'lihc', 'luad', 'lusc', 'meso', 'ov', 'paad', 'pcpg', 'prad', 'sarc', 'skcm', 'stad', 'tgct', 'thca', 'thym', 'ucec', 'ucs', 'uvm']


scaler = StandardScaler()
#scaler = MinMaxScaler()

model = Vae(config)
model = model.to(device)
print(model)

class MyDataset(Dataset):
    def  __init__(self, features, labels, is_train=True):
        self.features = features.astype(float)
        self.features = torch.Tensor(features)
        self.labels = torch.Tensor(labels.astype(float))

    def __getitem__(self, index):
        self.sample_features = self.features[index]
        self.sample_label = self.labels[index]
        return self.sample_features, self.sample_label

    def __len__(self):
        return len(self.features)

def dataset(type):
    train_features = pd.read_csv('D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/' + type + '_tcga_pan_can_atlas_2018' + '/train.csv', header=0, index_col=False, usecols=lambda column: column not in ['ENTITY_STABLE_ID']).to_numpy()
    val_features = pd.read_csv('D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/' + type + '_tcga_pan_can_atlas_2018' + '/val.csv', header=0, index_col=False, usecols=lambda column: column not in ['ENTITY_STABLE_ID']).to_numpy()
    test_features = pd.read_csv('D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/' + type + '_tcga_pan_can_atlas_2018' + '/test.csv', header=0, index_col=False, usecols=lambda column: column not in ['ENTITY_STABLE_ID']).to_numpy()

    print('Origin:')
    print(train_features.shape)


    train_features = np.transpose(train_features)
    val_features = np.transpose(val_features)
    test_features = np.transpose(test_features)


    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)



    #train_features = pd.DataFrame(train_features)
    print('stand:')
    print(train_features)

    features = [train_features, val_features, test_features]

    train = pd.read_csv('D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/' + type + '_tcga_pan_can_atlas_2018' + '/train.csv', usecols=lambda column: column not in ['ENTITY_STABLE_ID'])
    val = pd.read_csv('D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/' + type + '_tcga_pan_can_atlas_2018' + '/val.csv', usecols=lambda column: column not in ['ENTITY_STABLE_ID'])
    test = pd.read_csv('D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/' + type + '_tcga_pan_can_atlas_2018' + '/test.csv', usecols=lambda column: column not in ['ENTITY_STABLE_ID'])

    train_id = np.array(train.columns).tolist()
    val_id = np.array(val.columns).tolist()
    test_id = np.array(test.columns).tolist()

    y = pd.read_csv('D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/' + type + '_tcga_pan_can_atlas_2018' + '/Test_not_sort_data.csv')
    label = y['Label']
    Id = y['ID']

    train_label = []
    val_label = []
    test_label = []
    for trid in range(len(train_id)):
        for i in range(len(Id)):
            if train_id[trid] == Id[i]:
                train_label.append(label[i])
    for vaid in range(len(val_id)):
        for i in range(len(Id)):
            if val_id[vaid] == Id[i]:
                val_label.append(label[i])
    for teid in range(len(test_id)):
        for i in range(len(Id)):
            if test_id[teid] == Id[i]:
                test_label.append(label[i])

    train_labels = np.array(train_label)
    val_labels = np.array(val_label)
    test_labels = np.array(test_label)
    labels = [train_labels, val_labels, test_labels]

    num_features = train_features.shape[0]
    print('train features shape', train_features.shape)
    print('train labels shape', train_labels.shape)
    print('val features shape', val_features.shape)
    print('val labels shape', val_labels.shape)
    print('test features shape', test_features.shape)
    print('test labels shape', test_labels.shape)
    print(features[0])
    print()

    trainset = MyDataset(features[0], labels[0])
    valset = MyDataset(features[1], labels[1])
    testset = MyDataset(features[2], labels[2])
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    val_loader = DataLoader(valset, batch_size=64, shuffle=False)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False)
    return train_loader, val_loader, test_loader, trainset, valset, testset

reconstruction_function = nn.MSELoss(reduction='mean')

optimizer = optim.Adam(model.parameters(), lr=config.vae_learning_rate)

def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD


lamda_l1 = config.L1_regularization
best_val_loss = 999.999
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=config.scheduler_factor, patience=config.scheduler_patience, verbose=False)

def pre_train_model(num_epochs):
    print('VAE.....................................................')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        val_loss = 0
        num_batches = len(data_set[0])
        for batch_idx, data in enumerate(data_set[0]):
            model.train()
            starttime = datetime.datetime.now()
            mic_data, _ = data
            #mic_data = Variable(mic_data)
            mic_data = (mic_data.to(device) if torch.cuda.is_available() else mic_data)
            recon_batch, mu, logvar = model(mic_data)
            loss = loss_function(recon_batch, mic_data, mu, logvar)
            regularization_loss = 0.0
            for param in model.parameters():
                regularization_loss += torch.sum(abs(param))
            loss += lamda_l1 * regularization_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            train_loss += loss.item()
            '''
            if batch_idx % 100 == 0:
                endtime = datetime.datetime.now()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} time:{:.2f}s'.format(
                    epoch,
                    batch_idx * len(mic_data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(mic_data),
                    (endtime-starttime).seconds))
            '''
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(data_set[0].dataset)))
        #print(len(train_loader.dataset))


        print('val')
        for batch_idx, data in enumerate(data_set[1]):
            model.eval()
            starttime = datetime.datetime.now()
            mic_data, _ = data
            #mic_data = Variable(mic_data)
            mic_data = (mic_data.to(device) if torch.cuda.is_available() else mic_data)
            recon_batch, mu, logvar = model(mic_data)
            loss = loss_function(recon_batch, mic_data, mu, logvar)
            regularization_loss = 0.0
            for param in model.parameters():
                regularization_loss += torch.sum(abs(param))
            loss += lamda_l1 * regularization_loss
            val_loss += loss.item()
            if batch_idx % 100 == 0:
                '''
                endtime = datetime.datetime.now()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} time:{:.2f}s'.format(
                    epoch,
                    batch_idx * len(mic_data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(mic_data),
                    (endtime-starttime).seconds))
                '''
        print('====> Epoch: {} Average vloss: {:.4f}'.format(
            epoch, val_loss / len(data_set[1].dataset)))
        print('===========================================================')


    print('VAE DONE......................................................')
#torch.save(model.state_dict(), './vae.pth')

def pre_train_model_ready(num_epochs):
    print('VAE.....................................................')
    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        num_batches = len(data_set[0])
        for batch_idx, data in enumerate(data_set[0]):
            model.train()
            starttime = datetime.datetime.now()
            mic_data, _ = data
            #mic_data = Variable(mic_data)
            mic_data = (mic_data.to(device) if torch.cuda.is_available() else mic_data)
            recon_batch, mu, logvar = model(mic_data)
            loss = loss_function(recon_batch, mic_data, mu, logvar)

            regularization_loss = 0.0
            for param in model.parameters():
                regularization_loss += torch.sum(abs(param))
            loss += lamda_l1 * regularization_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            train_loss += loss.item()
            '''
            if batch_idx % 100 == 0:
                endtime = datetime.datetime.now()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} time:{:.2f}s'.format(
                    epoch,
                    batch_idx * len(mic_data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(mic_data),
                    (endtime-starttime).seconds))
            '''
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(data_set[0].dataset)))
        #print(len(train_loader.dataset))


        print('val')
        for batch_idx, data in enumerate(data_set[1]):
            model.eval()
            starttime = datetime.datetime.now()
            mic_data, _ = data
            #mic_data = Variable(mic_data)
            mic_data = (mic_data.to(device) if torch.cuda.is_available() else mic_data)
            recon_batch, mu, logvar = model(mic_data)
            loss = loss_function(recon_batch, mic_data, mu, logvar)
            regularization_loss = 0.0
            for param in model.parameters():
                regularization_loss += torch.sum(abs(param))
            loss += lamda_l1 * regularization_loss
            val_loss += loss.item()
            if batch_idx % 100 == 0:
                '''
                endtime = datetime.datetime.now()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} time:{:.2f}s'.format(
                    epoch,
                    batch_idx * len(mic_data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(mic_data),
                    (endtime-starttime).seconds))
                '''
        print('====> Epoch: {} Average vloss: {:.4f}'.format(
            epoch, val_loss / len(data_set[1].dataset)))
        print('===========================================================')
        if val_loss / len(data_set[1].dataset) < model.best_val_loss:
            model.best_val_loss = val_loss / len(data_set[1].dataset)
            save_path = 'D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/model/model_parameters.pth'
            torch.save(model.state_dict(), save_path)

data_set = dataset('blca')
num_epochs = config.number_epochs
pre_train_model(num_epochs)
save_path = 'D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/model/model_parameters.pth'
torch.save(model.state_dict(), save_path)

for type in cancer_type_list:
    save_path = 'D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/model/model_parameters.pth'
    model.load_state_dict(torch.load(save_path))
    data_set = dataset(type)
    pre_train_model_ready(num_epochs)

print(model.best_val_loss)
model.load_state_dict(torch.load(save_path))
print()