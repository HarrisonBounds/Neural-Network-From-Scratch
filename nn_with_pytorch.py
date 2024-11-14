import torch.nn as nn
import torch
import pandas as pd

class NeuralNet(nn.Module):
    def __init__(self, input_size, hl_size, output_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hl_size)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(hl_size, output_size)
        self.sigmoid2 = nn.Sigmoid()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.linear2(x)
        x = self.sigmoid2(x)
        return x
    

#Hyperparameters 
lr = 0.01
num_epochs = 500
hl_size = 5
outut_size = 1 #2 for multi-class cross entropy, 1 for MSE
label = "label"

#Alternatively, you could make a custom Dataloader class... didn't feel like it though
#Format Data
X_train = pd.read_csv("center_surround_train.csv")
X_test = pd.read_csv("center_surround_test.csv")
X_valid = pd.read_csv("center_surround_valid.csv")

#Seperate labels
y_train = X_train[label].values
y_test = X_test[label].values
y_valid = X_valid[label].values

#Drop labels
X_train = X_train.drop(label, axis=1).values
X_test = X_test.drop(label, axis=1).values
X_valid = X_valid.drop(label, axis=1).values

#Convert to tensors for numpy to use
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
X_valid = torch.tensor(X_valid, dtype=torch.float32)
y_valid = torch.tensor(y_valid, dtype=torch.float32)

#Build Network
ff = NeuralNet(X_train.shape[1], hl_size, outut_size)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(ff.parameters(), lr=lr)

#Use Stochastic Gradient Descent (only use one data example at a time)
for t in range(num_epochs):
    print(f"Epoch {t+1}\n===================================================")
    for i in range(len(X_train)):
        
        x = X_train[i]
        y = y_train[i]
        
        y_pred = ff(x)
        loss = loss_func(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Loss: {loss}")
    
#Evaluating the model
ff.eval() #Set the model to 'evaluation' mode

#Disable gradient calculation for testing
with torch.no_grad():
    correct = 0
    total = 0
    
    for i in range(len(X_valid)):
        x = X_valid[i]
        y = y_valid[i]
        
        pred = ff(x)
        
        #Clipping for MSE
        if pred >= 0.5:
            pred = 1
        elif pred < 0.5:
            pred = 0
        
        if pred == y:
            correct += 1
        
        total += 1
        
accuracy = correct / total
print(f"Accuracy on the validation set: {accuracy}")        

    

        
    
    
    
    









        
        
        