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
              batch_size_train=32,
              batch_size_test=32,
              normalize=True):

    def _get_index_train_test_path(split_num, dir_path, train = True):
        if train:
            return dir_path + "index_train_" + str(split_num) + ".txt"
        else:
            return dir_path + "index_test_" + str(split_num) + ".txt"

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

        return train_set, test_set, train_loader, test_loader, 0, 0, 0, 0

    else:

        data_path = "./data/"+data_name+"/data/data.txt"
        dir_path = "./data/"+data_name+"/data/"
        _INDEX_FEATURES_FILE = "./data/"+data_name+"/data/index_features.txt"
        _INDEX_TARGET_FILE = "./data/"+data_name+"/data/index_target.txt"
        #n_split = np.loadtxt("./data/"+data_name+"/data/n_splits.txt")
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

        ids_rem=[]
        for i in range(X_std.shape[0]):
            if X_std[i]==0:
                ids_rem.append(i)

        for i in ids_rem:
            X = np.concatenate((X[:,:i],X[:,i+1:]),axis=1)
            X_mean = np.concatenate((X_mean[:i],X_mean[i+1:]),axis=0)
            X_std = np.concatenate((X_std[:i],X_std[i+1:]),axis=0)

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

        train_loader=torch.utils.data.DataLoader(train_set, batch_size=batch_size_train,shuffle=False)
        test_loader=torch.utils.data.DataLoader(test_set, batch_size=batch_size_test,shuffle=False)


        return train_set, test_set, train_loader, test_loader, X_mean, X_std, y_mean, y_std


def prediction(network,testing_set,normalize=True):

    #network.eval()
    with torch.no_grad():
        if normalize:
            standard_pred = y_std*network(testing_set.data).squeeze(dim=1)+y_mean
        else:
            standard_pred = network(testing_set.data)

    rmse_standard_pred = np.mean((y_std*testing_set.targets.numpy()+y_mean - standard_pred.numpy())**2.)**0.5

    T = 10

    Yt_hat=[]
    for _ in range(T):
        Yt_hat.append(network(testing_set.data).squeeze(dim=1).data.numpy())
        #optimizer.step()
    Yt_hat=np.array(Yt_hat)
    if normalize:
        Yt_hat = Yt_hat * y_std + y_mean
    MC_pred = np.mean(Yt_hat, 0)
    rmse = np.mean((testing_set.targets.numpy()*y_std + y_mean - MC_pred)**2.)**0.5

    # We compute the test log-likelihood
    ll = (logsumexp(-0.5 * tau * (testing_set.targets.numpy()* y_std + y_mean - Yt_hat)**2., 0) - np.log(T)
        - 0.5*np.log(2*np.pi) + 0.5*np.log(tau))
    test_ll = np.mean(ll)

    # We are done!
    print("RMSE Standard pred")
    print(rmse_standard_pred)
    print("RMSE MC pred")
    print(rmse)
    print("LL")
    print(test_ll)

    return rmse_standard_pred, rmse, test_ll
