import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

def test1():
    dataFile29 = 'D://Desktop//pred_29.mat'
    dataFile31 = 'D://Desktop//pred_31.mat'
    data29 = scio.loadmat(dataFile29)
    data31 = scio.loadmat(dataFile31)
    print(data29)
    print(data31)

def test1205():
    data = np.loadtxt('D:\Desktop\QF15_ERP_human_P1.txt', skiprows=1, usecols=(1, 2))[63:126].T
    print(data)
    plt.figure()
    plt.plot(data[0], data[1])
    plt.show()

test1205()

