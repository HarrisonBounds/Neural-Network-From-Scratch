import pandas as pd
import numpy as np


class Node():
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))
        
    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases
        
        
class NeuralNetwork():
    def __init__(self, input_data, hl_size, lr):
        self.hl_size = hl_size
        self.data = np.array(input_data)
        print("data shape: ", self.data.shape)
        self.learning_rate = lr
        self.layer1 = Node(input_data.shape[0], self.hl_size)
        self.layer2 = Node(self.hl_size, 1) #One for binary classification
        print("Layer 1 weights shape: ", self.layer1.weights.shape)
        print("Layer 2 weights shape: ", self.layer2.weights.shape)
        
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        pass
    
    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
                            
    def forward_pass(self):
        self.z1 = self.layer1.forward(self.data.T)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.layer2.forward(self.a1)
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, y_true):
        dL_da2 = -2 * (self.a2 - y_true)
        da2_dz2 = (self.a2) * (1 - self.a2)
        dz2_dw2 =  self.a1
        
        
        dL_da2 = dL_da2.reshape(-1, 1)
        da2_dz2 = da2_dz2.reshape(-1, 1)
        dz2_dw2 = dz2_dw2.reshape(-1, 1)
        
        dL_dz2 = dL_da2 * da2_dz2
        dL_db2 = dL_dz2
        
        # print("dL_da2 shape: ", dL_da2.shape)
        # print("da2_dz2 shape: ", da2_dz2.shape)
        # print("dz2_dw2 shape: ", dz2_dw2.shape)
        
        dL_dw2 = np.dot(dz2_dw2, dL_da2 * da2_dz2)
        
        # print("dL_dw2 shape: ", dL_dw2.shape)
        
        dz2_da1 = self.layer2.weights
        da1_dz1 = (self.a1) * (1 - self.a1)
        dz1_dw1 = self.data
        
        dz2_da1 = dz2_da1.reshape(-1, 1)
        da1_dz1 = da1_dz1.reshape(-1, 1)
        dz1_dw1 = dz1_dw1.reshape(-1, 1)
        
        # print("\ndz2_da1 shape: ", dz2_da1.shape)
        # print("da1_dz1 shape: ", da1_dz1.shape)
        # print("dz1_dw1 shape: ", dz1_dw1.shape)
        
        dL_dw1 = np.dot(dz1_dw1, (dL_da2 * da2_dz2 * dz2_da1 * da1_dz1).T)
        
        dL_db1 = np.dot(dL_dz2, (dz2_da1 * da1_dz1).T)
        
        # print("dL_dw1 shape: ", dL_dw1.shape)
        
        self.layer1.weights -= self.learning_rate * dL_dw1
        self.layer2.weights -= self.learning_rate * dL_dw2
        
        self.layer2.biases -= self.learning_rate * dL_db2
        self.layer1.biases -= self.learning_rate * dL_db1

def main():
    hidden_layer_size = 5
    learning_rate = 0.01
    num_epochs = 1000
    #np.random.seed(11)
    y_preds = []
    correct = 0
    total = 0
    
    train_filename = "center_surround_train.csv"
    validate_filename = "center_surround_valid.csv"
    test_filename = "center_surround_test.csv"
    label_string = "label"

    X = pd.read_csv(train_filename)
    X_valid = pd.read_csv(validate_filename)
    V = pd.read_csv(validate_filename)
    T = pd.read_csv(test_filename)

    y = X[label_string].values
    X = X.drop(label_string, axis=1)
    
    y_valid = X_valid[label_string].values
    X_valid = X_valid.drop(label_string, axis=1)

    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    print("V shape: ", V.shape)
    print("T shape: ", T.shape)
    
    nn = NeuralNetwork(X.iloc[0], hidden_layer_size, learning_rate)
    # result = nn.forward_pass()
    # loss = nn.mse(y[0], result)
    # nn.backward(y[0])
    
    #Stochastic Gradient descent
    for i in range(num_epochs):
        print(f"Epoch {i}==================================================")
        for j in range(len(X)):  
            input_data = X.loc[j].values
            label = y[j]
            
            nn.data = input_data
            nn.label = label

            result = nn.forward_pass()
            loss = nn.mse(y[j], result)
            nn.backward(y[j])
            
        print(f"Loss: {loss}")
    
    #Evaulate model
    for j in range(len(X_valid)):
        input_data = X_valid.iloc[j].values
        nn.data = input_data
        
        y_pred = nn.forward_pass()
        
        if y_pred >= 0.5:
            y_pred = 1
        elif y_pred < 0.5:
            y_pred = 0
            
        if y_pred == y_valid[j]:
            correct += 1
            
        total += 1
        
    accuracy = correct / total
    
    print(f"Accuracy: ", accuracy)
        
    
if __name__ == '__main__':
    main()