import numpy as np
import csv
from matplotlib import pyplot as plt
from numpy import random
from numpy.random.mtrand import uniform


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    """
    TODO: implement this function.

    INPUT: 
        filename - a string representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    dataset = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:            
            dataset.append([float(row['BODYFAT']), float(row['DENSITY']), float(row['AGE']), float(row['WEIGHT']), float(row['HEIGHT']), float(row['ADIPOSITY']), float(row['NECK']), float(row['CHEST']), float(row['ABDOMEN']), float(row['HIP']), float(row['THIGH']), float(row['KNEE']), float(row['ANKLE']), float(row['BICEPS']), float(row['FOREARM']), float(row['WRIST'])])
    return np.array(dataset)


def print_stats(dataset, col):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on. 
                  For example, 1 refers to density.

    RETURNS:
        None
    """
    length = len(dataset)
    print(length)

    sum = 0
    for row in dataset:
        sum += row[col]
    
    mean = sum/length
    mean_print = "{:.2f}".format(mean)
    print(mean_print)

    temp = 0
    for row in dataset:
        temp += pow((row[col] - mean), 2)

    temp = temp/(length - 1)
    std_dev = pow(temp, 0.5)
    std_dev_print = "{:.2f}".format(std_dev)
    print(std_dev_print)
    pass


def regression(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model
    """
    mse = 0
    for row in dataset:
        mse_i = betas[0] - row[0]
        for i in range(0, len(cols)):
            mse_i += row[cols[i]]*betas[i + 1]
        mse += pow(mse_i, 2)

    mse = mse/len(dataset)
    return mse


def gradient_descent(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        An 1D array of gradients
    """
    grads = [0]*len(betas)

    for row in dataset:
        temp = betas[0] - row[0]
        for i in range(0, len(cols)):
            temp += row[cols[i]]*betas[i + 1]

        grads[0] += temp
        for i in range(0, len(cols)):
            grads[i + 1] += (temp)*row[cols[i]]

    grads = np.array(grads)
    grads = np.multiply(grads, 2)
    grads = np.divide(grads, len(dataset))
    return grads


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """
    # setting up the 2-D matrix containing all betas
    itr_betas = [[0.0]*(T+1)]*(len(betas))
    itr_betas = np.array(itr_betas)
    for i in range(0, len(betas)):
        itr_betas[i][0] = betas[i]

    #computing all betas
    for i in range(1, T + 1):
        for j in range(0, len(betas)):
            itr_betas[j][i] = itr_betas[j][i - 1] - eta*(gradient_descent(dataset, cols, itr_betas[:,i-1])[j])

    # printing out
    for i in range(1, T+1):
        print_str = str(i) + " " + str(round(regression(dataset, cols, itr_betas[:,i]), 2)) + " " + str(round(itr_betas[0][i], 2)) + " "
        for j in range(1, len(betas)):
            print_str += str(round(itr_betas[j][i], 2)) + " "
        print(print_str)

    pass


def compute_betas(dataset, cols):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    betas = []
    Y = dataset[:,0]
    X = []

    for i in range(0, len(dataset)):
        row_i = [1]
        for j in range(0, len(cols)):
            row_i.append(dataset[i][cols[j]])
        X.append(row_i)
    X = np.array(X)
    X_t = np.transpose(X)
    
    betas = np.matmul(np.linalg.inv(np.matmul(X_t, X)), np.dot(X_t, Y))
    mse = regression(dataset, cols, betas)

    return (mse, *betas)


def predict(dataset, cols, features):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    betas = compute_betas(dataset, cols)
    result = betas[1]

    for i in range(0, len(features)):
        result += betas[i + 2]*features[i]

    return result


def synthetic_datasets(betas, alphas, X, sigma):
    """
    TODO: implement this function.

    Input:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise

    RETURNS:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """
    lin = np.array([[0.0, 0.0]]*(len(X)))
    quad = np.array([[0.0, 0.0]]*(len(X)))

    for i in range(len(X)):
        lin[i][1] = X[i][0]
        lin[i][0] = betas[0] + betas[1]*lin[i][1] + np.random.normal(0.0, sigma)
    
    for i in range(len(X)):
        quad[i][1] = X[i][0]
        quad[i][0] = alphas[0] + alphas[1]*quad[i][1]*quad[i][1] + np.random.normal(0.0, sigma)

    return (lin, quad)


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    # TODO: Generate datasets and plot an MSE-sigma graph
    X = []
    for i in range(1000):
        X.append([random.uniform(-100, 100)])
    X = np.array(X)

    betas = [random.uniform(), random.uniform()]
    alphas = [random.uniform(), random.uniform()]
    sigmas = [1/10000, 1/1000, 1/100, 1/10, 1, 10, 100, 1000, 10000, 100000]
    betas = np.array(betas)
    alphas = np.array(alphas)
    sigmas = np.array(sigmas)

    lin_datasets = []
    quad_datasets = []
    for sigma in sigmas:
        lin_datasets.append(compute_betas(synthetic_datasets(betas, alphas, X, sigma)[0], cols=[1])[0])
        quad_datasets.append(compute_betas(synthetic_datasets(betas, alphas, X, sigma)[1], cols=[1])[0])

    lin_datasets = np.array(lin_datasets)
    quad_datasets = np.array(quad_datasets)

    # plotting
    plt.figure()
    plt.plot(sigmas, lin_datasets, label = 'MSE of Linear Dataset', marker="o")
    plt.plot(sigmas, quad_datasets, label = 'MSE of Quadratic Dataset', marker="o")
    
    plt.xlabel('Standard Deviation of Error Term')
    plt.ylabel('MSE of Trained Model')

    plt.xscale("log")
    plt.yscale("log")

    plt.legend()
    # Display a figure
    plt.savefig('mse.pdf')
    


if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()
