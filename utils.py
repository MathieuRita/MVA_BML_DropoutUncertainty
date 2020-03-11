import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torch.optim as optim
from sklearn.model_selection import train_test_split

class Dataset(torch.utils.data.Dataset):

    """
    Dataset class for pytorch
    """

    def __init__(self,X,y):

        super(Dataset, self).__init__()

        self.data=torch.Tensor(X)
        self.targets=torch.Tensor(y)
        self.n_samples=X.shape[0]

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        X = self.data[index]
        y = self.targets[index]

        return X,y


def load_data(data_name="MNIST",
              batch_size_train=64,
              batch_size_test=1000,
              normalize=True):

    if data_name=="MNIST":
        train_set = torchvision.datasets.MNIST('./data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                     torchvision.transforms.ToTensor(),
                                     torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                                   ]))

        train_set.data=train_set.data.flatten(start_dim=1, end_dim=2)

        test_set = torchvision.datasets.MNIST('./data', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                     torchvision.transforms.ToTensor(),
                                     torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                                   ]))

        test_set.data = test_set.data.flatten(start_dim=1, end_dim=2)

        train_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST('./data/', train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))
                                     ])),
          batch_size=batch_size_train, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST('./data/', train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))
                                     ])),
          batch_size=batch_size_test, shuffle=True)

    else:

        data_path = "./data/"+data_name+"/data/data.txt"
        dir_path = "./data/"+data_name+"/data/"
        _INDEX_FEATURES_FILE = "./data/"+data_name+"/data/index_features.txt"
        _INDEX_TARGET_FILE = "./data/"+data_name+"/data/index_target.txt"
        n_split = np.loadtxt("./data/"+data_name+"/data/n_splits.txt")
        index_train = np.loadtxt(_get_index_train_test_path(0, dir_path=dir_path, train=True))
        index_test = np.loadtxt(_get_index_train_test_path(0, dir_path=dir_path, train=False))

        data = np.loadtxt(data_path)
        index_features = np.loadtxt(_INDEX_FEATURES_FILE)
        index_target = np.loadtxt(_INDEX_TARGET_FILE)

        X = data[ : , [int(i) for i in index_features.tolist()] ]
        y = data[ : , int(index_target.tolist()) ]

        X_mean=np.mean(X,axis=0)
        X_std=np.std(X,axis=0)
        y_mean=np.mean(y)
        y_std=np.std(y)

        if normalize:
            X=(X-X_mean)/X_std
            y=(y-y_mean)/y_std

        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = X[ [int(i) for i in index_train.tolist()] ]
        y_train = y[ [int(i) for i in index_train.tolist()] ]

        X_test = X[ [int(i) for i in index_test.tolist()] ]
        y_test = y[ [int(i) for i in index_test.tolist()] ]

        train_set = Dataset(X=X_train, y=y_train)
        test_set = Dataset(X=X_test, y=y_test)

        train_loader=torch.utils.data.DataLoader(train_set, batch_size=batch_size_train,shuffle=True)
        test_loader=torch.utils.data.DataLoader(test_set, batch_size=batch_size_test,shuffle=True)


    return train_set, test_set, train_loader, test_loader, X_mean, X_std, y_mean, y_std



def _get_index_train_test_path(split_num, dir_path, train = True):
    """
       Method to generate the path containing the training/test split for the given
       split number (generally from 1 to 20).
       @param split_num      Split number for which the data has to be generated
       @param train          Is true if the data is training data. Else false.
       @return path          Path of the file containing the requried data
    """
    if train:
        return dir_path + "index_train_" + str(split_num) + ".txt"
    else:
        return dir_path + "index_test_" + str(split_num) + ".txt"
