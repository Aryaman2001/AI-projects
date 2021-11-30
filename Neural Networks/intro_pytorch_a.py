import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
torch.utils.data.DataLoader

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.
    

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    if (training == True):
        train_set = datasets.MNIST('./data', train=True, download=True,
                       transform=custom_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size = 50, shuffle=False)
        return train_loader
    else:
        test_set = datasets.MNIST('./data', train=False,
                       transform=custom_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size = 50, shuffle=False)
        return test_loader
   
    
def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128, bias=True),
        nn.ReLU(),
        nn.Linear(128, 64, bias=True),
        nn.ReLU(),
        nn.Linear(64, 10, bias=True),
    )
    return model


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(T):  # loop over the dataset multiple times
        model.train()
        loss = 0.0
        accuracy = 0
        sum = 0
        for batch, (image, label) in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            #.eq, argmax,
            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(image)
            # print(len(outputs))
            loss = criterion(outputs, label)
            loss.backward()
            opt.step()

            # print statistics
            loss += loss.item()
            _, prediction = torch.max(outputs, dim=1)
            accuracy += torch.sum(prediction == label).item()
            sum += label.size(0) 
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #         (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0
        epoch_print = str(epoch)
        accuracy_print = "{}/{}({:.2f}%)".format(accuracy, sum, 100*accuracy/sum)
        loss_print = "{:.3f}".format(loss.item())
        print("​Train Epoch: " + epoch_print + " Accuracy: " + accuracy_print + " Loss: " + loss_print)

def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    # criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        accuracy = 0
        sum = 0
        for batch, (image, label) in enumerate(test_loader):
            outputs = model(image)
            loss = criterion(outputs, label)
            _, prediction = torch.max(outputs, dim=1)
            accuracy += torch.sum(prediction == label).item()
            sum += label.size(0) 
               
        accuracy_print = "{:.2f}%".format(100*accuracy/sum)
        if show_loss:
            loss_print = "{:.4f}".format(loss.item())
            print("Average Loss: " + loss_print)
        print("Accuracy: " + accuracy_print)


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    img = test_images[index]
    outputs = model(img)
    prob = F.softmax(outputs, dim = 1)
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    probRetList = []

    for i in range(len(class_names)):
        probRetList.append([prob[0][i], class_names[i]])

    probRetList = sorted(probRetList, reverse = True)

    for i in range(3):
        number_print = str(probRetList[i][1])
        probability_print = '{:.2f}%'.format(100 * probRetList[i][0])
        print(number_print + ": " + probability_print)


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()

# train_loader = get_data_loader()
# test_loader = get_data_loader(training=False)
# model = build_model()
# evaluate_model(model, test_loader, criterion, show_loss = True)
# predset, _ = iter(get_data_loader()).next()
# predict_label(model, predset, 1)
