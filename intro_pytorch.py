import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    if(training):
        train_set = datasets.FashionMNIST('./data', train=True, download=True, transform=custom_transform)
    else:
        train_set = datasets.FashionMNIST('./data', train=False, transform=custom_transform)
    return torch.utils.data.DataLoader(train_set, batch_size=64)



def build_model():
    """

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28,128), 
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,10)
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

    #criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    for epoch in range(T):
        running_loss = 0.0
        correct = 0
        total = 0
        batch_size = 0
        for i, batch in enumerate(train_loader,0):
            image, label = batch

            opt.zero_grad()
            
            output = model(image)
            loss = criterion(output, label)
            _,predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            loss.backward()
            opt.step()
            running_loss += loss.item()
            if(batch_size == 0):
                batch_size = label.size(0)
        print(f'Train Epoch: {epoch} Accuracy: {correct}/{total}({correct/total * 100:.2f}%) Loss: {running_loss * batch_size/total:.3f}')



    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.eval()
    running_loss = 0.0
    batch_size = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for image, label in test_loader:

            opt.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            _,predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            running_loss += loss.item()
            if(batch_size==0):
                batch_size = label.size(0)
    if(show_loss):
        print(f'Average loss: {running_loss * batch_size/total:.4f}')
    print(f"Accuracy: {correct/total * 100:.2f}%")



    


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

    prob = F.softmax(model(test_images[index]),1)
    Dict = dict()
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt'
                ,'Sneaker','Bag','Ankle Boot']
    i = 0
    for p in prob.tolist()[0]:
        Dict[p] = class_names[i]
        i += 1
    count = 0
    for i in sorted(Dict.keys(),reverse=True):
        if(count == 3):
            break
        print(f'{Dict[i]}: {i*100:.2f}%')
        count += 1




if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    print(type(train_loader))
    print(train_loader.dataset)

    model = build_model()
    print(model)
    criterion = nn.CrossEntropyLoss()
    train_model(model,train_loader,criterion,5)
    evaluate_model(model, train_loader, criterion, show_loss = True)
    pred_set, _ = next(iter(test_loader))
    predict_label(model, pred_set, 1)