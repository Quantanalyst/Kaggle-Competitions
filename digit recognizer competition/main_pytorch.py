
"""
Title: Digit Recognizer (Kaggle Competition)
Created on Thu Jan 23 16:04:27 2020

@author: Saeed Mohajeryami, PhD

"""

# Import Libraries
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Prepare Dataset
# load data
train = pd.read_csv("train.csv")

# split data into features and label dataset
X_train = train.iloc[:,1:].values/255  #read, convert to array, normalization
y_train = train.iloc[:,0].values       #labels(numbers from 0 to 9)

# split the data to train/evaluation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# create feature and label tensor for training. 
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for evaluation.
X_val_tensor = torch.from_numpy(X_val)
y_val_tensor = torch.from_numpy(y_val).type(torch.LongTensor)

# batch_size, epoch and iteration
batch_size = 100
n_iters = 10000
num_epochs = n_iters / (len(X_train) / batch_size)
num_epochs = int(num_epochs)

# Pytorch train and evaluation sets
train_torchtensor = torch.utils.data.TensorDataset(X_train_tensor,y_train_tensor)
eval_torchtensor = torch.utils.data.TensorDataset(X_val_tensor,y_val_tensor)

# data loader
train_loader = torch.utils.data.DataLoader(train_torchtensor, batch_size = batch_size, shuffle = False)
eval_loader = torch.utils.data.DataLoader(eval_torchtensor, batch_size = batch_size, shuffle = False)

# visualize one of the images in data set
plt.imshow(X_train[10].reshape(28,28))
plt.axis("off")
plt.title(str(y_train[10]))
plt.savefig('graph.png')
plt.show()




### ------------------ Classification and Prediction ------------------------
### ---------------------- Logistic Regression ------------------------------
# Create Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__() ## inherit from nn.Module
        # Linear part
        self.linear = nn.Linear(input_dim, output_dim)
        # There should be logistic function right?
        # However logistic function in pytorch is in loss function
        # So actually we do not forget to put it, it is only at next parts
    
    def forward(self, x):
        out = self.linear(x)
        return out

# Instantiate Model Class
input_dim = 28*28 # size of image px*px
output_dim = 10  # labels 0,1,2,3,4,5,6,7,8,9

# create logistic regression model
model = LogisticRegressionModel(input_dim, output_dim)

# Cross Entropy Loss  
error = nn.CrossEntropyLoss()

# SGD Optimizer 
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Traning the Model
count = 0
loss_list = []
iteration_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): ##unpack each batch
        
        # Define variables
        train = Variable(images.view(-1, 28*28)).type(torch.FloatTensor)
        labels = Variable(labels)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(train)
        
        # Calculate softmax and cross entropy loss
        loss = error(outputs, labels)
        
        # Calculate gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        count += 1
        
        # Prediction
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Predict test dataset
            for images, labels in eval_loader: 
                test = Variable(images.view(-1, 28*28)).type(torch.FloatTensor)
                
                # Forward propagation
                outputs = model(test)
                
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += len(labels)
                
                # Total correct predictions
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
        if count % 500 == 0:
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {}%'.format(count, loss.data, accuracy))
            

ll = []
for i in range(len(loss_list)):
    ll.append(loss_list[i-1][0])

# visualization
plt.plot(iteration_list,ll)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("Logistic Regression: Loss vs Number of iteration")
plt.show()

### ---------------------- Artificial NN ------------------------------
# Logistic regression is good at classification but when complexity
# (non linearity) increases, the accuracy of model decreases. So, we need to
# increase complexity of model and add more non linear functions as hidden layer.
# With more hidden layers, the model can adapt better. As a result accuracy
# increase.
# Create ANN Model
class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANNModel, self).__init__()
        # Linear function 1: 784 --> 100
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity 1
        self.relu1 = nn.ReLU()
        
        # Linear function 2: 100 --> 100
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.tanh2 = nn.Tanh()
        
        # Linear function 3: 100 --> 100
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 3
        self.elu3 = nn.ELU()
        
        # Linear function 4 (readout): 100 --> 10
        self.fc4 = nn.Linear(hidden_dim, output_dim)  
    
    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)
        
        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.tanh2(out)
        
        # Linear function 2
        out = self.fc3(out)
        # Non-linearity 2
        out = self.elu3(out)
        
        # Linear function 4 (readout)
        out = self.fc4(out)
        return out

# instantiate ANN
input_dim = 28*28
hidden_dim = 150 #hidden layer dim is one of the hyper parameter and it should be chosen and tuned. For now I only say 150 there is no reason.
output_dim = 10

# Create ANN
model = ANNModel(input_dim, hidden_dim, output_dim)

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# ANN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        train = Variable(images.view(-1, 28*28)).type(torch.FloatTensor)
        labels = Variable(labels)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(train)
        
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        
        # Calculating gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        count += 1
        
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Predict test dataset
            for images, labels in eval_loader:

                test = Variable(images.view(-1, 28*28)).type(torch.FloatTensor)
                
                # Forward propagation
                outputs = model(test)
                
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += len(labels)

                # Total correct predictions
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            if count % 500 == 0:
                # Print Loss
                print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data[0], accuracy))


ll = []
for i in range(len(loss_list)):
    ll.append(loss_list[i-1][0])
    
# visualization loss 
plt.plot(iteration_list,ll)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("ANN: Loss vs Number of iteration")
plt.show()

# visualization accuracy 
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("ANN: Accuracy vs Number of iteration")
plt.show()


### ---------------------- Convolutional NN ------------------------------
# CNN is well adapted to classify images.

# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected 1
        self.fc1 = nn.Linear(32 * 4 * 4, 10) 
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        
        # Max pool 1
        out = self.maxpool1(out)
        
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)
        
        # Max pool 2 
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)
        
        return out

# batch_size, epoch and iteration
batch_size = 100
n_iters = 2500
num_epochs = n_iters / (len(X_train) / batch_size)
num_epochs = int(num_epochs)

# Pytorch train and test sets
train_torchtensor = torch.utils.data.TensorDataset(X_train_tensor,y_train_tensor)
eval_torchtensor = torch.utils.data.TensorDataset(X_val_tensor,y_val_tensor)

# data loader
train_loader = torch.utils.data.DataLoader(train_torchtensor, batch_size = batch_size, shuffle = False)
eval_loader = torch.utils.data.DataLoader(eval_torchtensor, batch_size = batch_size, shuffle = False)


# Create ANN
model = CNNModel()

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        train = Variable(images.view(100,1,28,28)).type(torch.FloatTensor)
        labels = Variable(labels)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(train)
        
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        
        # Calculating gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        count += 1
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in eval_loader:
                
                test = Variable(images.view(100,1,28,28)).type(torch.FloatTensor)
                
                # Forward propagation
                outputs = model(test)
                
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += len(labels)
                
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            if count % 500 == 0:
                # Print Loss
                print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data[0], accuracy))
                
ll = []
for i in range(len(loss_list)):
    ll.append(loss_list[i-1][0])                
# visualization loss 
plt.plot(iteration_list,ll)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("CNN: Loss vs Number of iteration")
plt.show()

# visualization accuracy 
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("CNN: Accuracy vs Number of iteration")
plt.show()



# ----------------------------------Results and Submission ------------------
test = pd.read_csv("test.csv")
index = np.array(test.index + 1)
test = test.values/255
test_tensor = torch.from_numpy(test)
index_tensor = torch.from_numpy(index)
#test_tensor = test_tensor.contiguous()
#test_pytorch = Variable(test_tensor.view(-1,1,28,28)).type(torch.FloatTensor)
test_torchtensor = torch.utils.data.TensorDataset(test_tensor,index_tensor)

# data loader
batch_size = 100
test_loader = torch.utils.data.DataLoader(test_torchtensor, batch_size = batch_size, shuffle = False)

# Iterate through test dataset and predict
prediction = []
for images,_ in test_loader:
    
    test = Variable(images.view(100,1,28,28)).type(torch.FloatTensor)
    
    # Forward propagation
    outputs = model(test)
    
    # Get predictions from the maximum value
    predicted = torch.max(outputs.data, 1)[1]
    
    pred_list = [value for value in predicted]
    
    prediction = prediction + pred_list
    

results = pd.Series(prediction,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("pytorchSubmission.csv",index=False)