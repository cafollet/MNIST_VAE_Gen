import numpy as np


# Defining a Data Class
class Data:
    def __init__(self):
        x_train = np.delete(np.loadtxt("data/mnist_train.csv",
                                       skiprows=1, delimiter=','), 0, 1)  # Image training dataset
        y_train = np.loadtxt("data/mnist_train.csv",
                             skiprows=1, usecols=0, delimiter=',')  # Label training dataset

        label_train = y_train

        self.x_train = x_train.astype(np.float32)
        self.y_train = y_train.astype(np.float32)
        self.label_train = np.asarray(label_train)
