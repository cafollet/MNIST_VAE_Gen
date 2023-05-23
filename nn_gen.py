import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import math

conv_stride = 2  # convolution kernel stride
conv_chans1 = 8  # convolutional channel 1 output
conv_chans2 = 16  # convolutional channel 2 output
conv_chans3 = 32  # convolutional channel 3 output
conv_size1 = (28 // conv_stride)  # convolution single channel size output
conv_out_size1 = conv_chans1 * (conv_size1 ** 2)
conv_size2 = (math.floor(conv_size1 / conv_stride))
conv_out_size2 = conv_chans2 * (conv_size2 ** 2)
conv_size3 = (math.floor(conv_size2 / conv_stride))
conv_out_size3 = conv_chans3 * (conv_size3 ** 2)  # convolutions total size output
class Encoder(nn.Module):
    def __init__(self, latent_dims, hidden_layer):
        """
        a network of 1 convolutional layer followed by 3 linear layers, 2 of which connect to the latent dimension,
        and define the average and variance of the generating model
        :param latent_dims: scalar latent dimension size
        :param hidden_layer: scalar hidden layer size
        """
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, conv_chans1, 3, stride=conv_stride, padding=1)
        self.conv2 = nn.Conv2d(conv_chans1, conv_chans2, 3, stride=conv_stride, padding=1)
        self.batch = nn.BatchNorm2d(conv_chans2)
        self.conv3 = nn.Conv2d(conv_chans2, conv_chans3, 3, stride=conv_stride, padding=0)
        self.fc1 = nn.Linear(conv_out_size3, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, latent_dims)
        self.fc3 = nn.Linear(hidden_layer, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        """
        defines the forward function of the encoder
        :param x: a 3 or 4-dimensional pytorch tensor with each dimension
                  being size (batch size:optional, number of channels, height, width) in that order
        :return: the distribution generated from the latent space, z
        """
        x = func.relu(self.conv1(x))
        x = func.relu(self.batch(self.conv2(x)))
        x = func.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = func.relu(self.fc1(x))
        mu = self.fc2(x)
        log_var = self.fc3(x)
        z = mu + torch.exp(log_var) * self.N.sample(mu.shape)
        kl_loss = -0.5 * torch.mean(1 + log_var - torch.square(mu) - torch.exp(log_var))
        self.kl = kl_loss
        return z

class Net(nn.Module):
    """
    Creating a Neural Net class with two fully connected layers,
 and non-linear activation functions reLU and logSoftmax
    """

    def __init__(self, n_input, n_hidden):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)  # input to hidden layer
        self.fc2 = nn.Linear(n_hidden, 10)  # hidden to output layer

    # Feedforward:
    def forward(self, x):  # h: hidden layer; vector, y: output layer;vector of length 5
        """
        defines a function to run a dataset through the neural network
        """
        h = func.relu(self.fc1(x))
        y = func.log_softmax(self.fc2(h), dim=1)
        """ Need softmax, which outputs multiple classes, but NLLLoss only accepts log Softmax, 
        so this is what we will use"""
        return y

    # Backpropagation:
    def backprop(self, data, loss, optimizer, lbatch, rbatch):
        """
        defines the function generate a loss value and accuracy from a dataset with associating labels,
        and train the network using gradient descent
        """
        self.train()
        inputs = torch.from_numpy(data.x_train[lbatch:rbatch])
        targets = torch.from_numpy(data.label_train[lbatch:rbatch]).long()
        obj_val = loss(self.forward(inputs), targets)
        max_vals, max_indexes = self.forward(inputs).max(dim=1)

        """ 
        Using the knowledge from our labels as vectors, we know the highest probability of 
        one entry corresponds with the index number  of the entry (times 2) being the most probable target, 
        so we make it our target"""
        pred_targets = max_indexes

        """
        How many predicted targets are correct over
        the number of targets being analysed (the batch size), which gives the averaged accuracy"""
        acc = (targets == pred_targets).sum() / targets.size(0)
        train_acc = acc.item() # Test Accuracy
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item(), train_acc, pred_targets

    def test(self, x):
        max_vals, max_index = self.forward(x).max(dim=1)
        label = max_index.item()
        return label

class Decoder(nn.Module):
    """ Decodes the network by applying the same layers and activation functions as the encoder class, \
    but in the opposite direction
    """

    def __init__(self, latent_dims, hidden_layer):
        super().__init__()
        self.fc4 = nn.Linear(latent_dims, hidden_layer)
        self.fc6 = nn.Linear(hidden_layer, conv_out_size3)
        self.rev_conv3 = nn.ConvTranspose2d(conv_chans3, conv_chans2, 3, stride=conv_stride, output_padding=0)
        self.rev_batch2 = nn.BatchNorm2d(conv_chans2)
        self.rev_conv2 = nn.ConvTranspose2d(conv_chans2, conv_chans1, 3, stride=conv_stride, padding=1, output_padding=1)
        self.rev_batch1 = nn.BatchNorm2d(conv_chans1)
        self.rev_conv1 = nn.ConvTranspose2d(conv_chans1, 1, 3, stride=conv_stride, padding=1, output_padding=1)


    def forward(self, x):
        """
        defines the forward function of the decoder, which is the encoder forward process in the opposite direction
        :param x: a 3 or 4-dimensional pytorch tensor with each dimension
                  being size (batch size:optional, number of channels, height, width) in that order
        :return:
        """
        x_1 = self.fc4(x)
        x_2 = self.fc6(x_1)
        unflatten = nn.Unflatten(1, (conv_chans3, conv_size3, conv_size3)) # opposite process of flatten in
        x_3 = unflatten(x_2)
        x_4 = self.rev_conv3(x_3)
        x_5 = self.rev_conv2(self.rev_batch2(x_4))
        x_6 = self.rev_conv1(self.rev_batch1(x_5))
        x_7 = torch.remainder(torch.abs(x_6), 256)
        return x_7

class VariationalAutoencoder(nn.Module):
    """
    Combines the two networks made above, creating one VAE network
    """
    def __init__(self, latent_dims, hidden):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dims, hidden)
        self.decoder = Decoder(latent_dims, hidden)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
