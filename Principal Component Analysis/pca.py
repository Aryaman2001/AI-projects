
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # TODO: add your code here
    x = np.load(filename) 
    x = x - np.mean(x, axis=0)
    return x


def get_covariance(dataset):
    # TODO: add your code here
    matrix = np.dot(np.transpose(dataset), dataset)
    return np.true_divide(matrix, len(dataset) - 1)


def get_eig(S, m):
    # TODO: add your code here
    eigVal, eigVec = eigh(S, eigvals=(len(S) - m, len(S) - 1)) 

    # sorting in descending
    for i  in range(0, len(eigVec)):
        eigVec[i] = np.flip(eigVec[i])

    eigVal = np.flip(eigVal)
    eigVal = np.diag(eigVal)
  
    return eigVal, eigVec

def get_eig_perc(S, perc):
    # TODO: add your code here

    # get all eigenvalues and vectors
    eigVal, eigVec = eigh(S)
    eigValSum = sum(eigVal)     
    percentage = eigValSum * perc

    # eigen values and vectors which explain given perc
    size = len([element for element in eigVal if element > percentage])
    retEigVal, retEigVec = eigh(S, eigvals=(len(S) - size, len(S) - 1))

    # return them in descending order
    for i  in range(0, len(retEigVec)):
        retEigVec[i] = np.flip(retEigVec[i])

    retEigVal = np.flip(retEigVal)
    retEigVal = np.diag(retEigVal)
  
    return retEigVal, retEigVec

def project_image(img, U):
    # TODO: add your code here
    return np.dot(U, np.dot(np.transpose(U), img))


def display_image(orig, proj):
    # TODO: add your code here

    # reshape image
    orig = np.reshape(orig, (32, 32), order='F')
    proj = np.reshape(proj, (32, 32), order='F')

    # create subplot
    fig, (ax1, ax2) = plt.subplots(1,2)

    # set title for subplots
    ax1.set_title("Original")
    ax2.set_title("Projection")

    # render image in subplots
    origFig = ax1.imshow(orig, aspect='equal')
    projFig = ax2.imshow(proj, aspect='equal')

    # colorbar
    fig.colorbar(origFig, ax=ax1)
    fig.colorbar(projFig, ax=ax2)

    # render
    plt.show()
    return