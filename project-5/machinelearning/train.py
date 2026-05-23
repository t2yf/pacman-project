from torch import no_grad
from torch.utils.data import DataLoader


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


"""
##################
### QUESTION 1 ###
##################
"""


def train_perceptron(model, dataset):
    """
    Train the perceptron until convergence.
    You can iterate through DataLoader in order to 
    retrieve all the batches you need to train on.

    Each sample in the dataloader is in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.
    """
    with no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        converged = False
        while not converged:    
            converged = True    
            for batch in dataloader:
                x = batch['x'].squeeze() # remover a dimensão do batch
                y = batch['label'].item() # escalar
                pred = model.get_prediction(x)
                if pred != y:
                    model.w += y*x
                    converged = False



def train_regression(model, dataset):
    """
    Trains the model.

    In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
    batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

    Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.

    Inputs:
        model: Pytorch model to use
        dataset: a PyTorch dataset object containing data to be trained on
        
    """
    epochs = 300
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            x = batch['x']
            y = batch['label']

            optimizer.zero_grad()
            pred = model(x)

            loss = regression_loss(pred, y)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        #early stopping se loss chegar a esse valor
        if avg_loss < 0.01:
            break
    


def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """
    model.train()
    epochs = 30
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    for epoch in range(epochs):
        for batch in dataloader:
            x = batch['x']
            y = batch['label']

            optimizer.zero_grad()
            pred = model(x)

            loss = digitclassifier_loss(pred, y)
            loss.backward()

            optimizer.step()
        
        if dataset.get_validation_accuracy() > 0.98:
            break



def train_languageid(model, dataset):
    """
    Trains the model.

    Note that when you iterate through dataloader, each batch will returned as its own vector in the form
    (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
    get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
    that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
    as follows:

    movedim(input_vector, initial_dimension_position, final_dimension_position)

    For more information, look at the pytorch documentation of torch.movedim()
    """
    model.train()
    epochs = 20
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr = 0.01)

    for epoch in range(epochs):
        for batch in dataloader:
            xs = batch['x']
            y = batch['label']

            xs = movedim(xs, 0, 1)

            optimizer.zero_grad()
            pred = model(xs)

            loss = languageid_loss(pred, y)
            loss.backward()

            optimizer.step()
        


def Train_DigitConvolution(model, dataset):
    """
    Trains the model.
    """
    model.train()
    epochs = 200
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    for epoch in range(epochs):

        for batch in dataloader:

            x = batch['x']
            y = batch['label']

            optimizer.zero_grad()

            pred = model(x)

            loss = digitconvolution_Loss(pred, y)

            loss.backward()

            optimizer.step()

        if dataset.get_validation_accuracy() > 0.80:
            break