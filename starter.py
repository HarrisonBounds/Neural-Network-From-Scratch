import pandas as pd
import numpy as np

train_filename = "center_surround_train.csv"
validate_filename = "center_surround_valid.csv"
test_filename = "center_surround_test.csv"
label_string = "label"

X = pd.read_csv(train_filename)
V = pd.read_csv(validate_filename)
T = pd.read_csv(test_filename)

y = X[label_string].values
X = X.drop(label_string, axis=1)

print("X shape: ", X.shape)
print("y shape: ", y.shape)
print("V shape: ", V.shape)
print("T shape: ", T.shape)

def mse_derivative(y_pred):
    return 

class Node():
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros(output_size)
        
    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases
        
        
class NeuralNetwork():
    def __init__(self, input_data, label, hl_size, lr):
        self.hl_size = hl_size
        self.data = input_data
        self.label = label
        self.learning_rate = lr
        self.layer1 = Node(input_data.shape[0], self.hl_size)
        self.layer2 = Node(self.hl_size, 1) #One for binary classification
        
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        pass
    
    def mse(self, y_pred):
        return np.mean((self.label - y_pred)**2)
                            
    def forward_pass(self):
        self.z1 = self.layer1.forward(self.data.T)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.layer2.forward(self.a1)
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self):
        dL_da2 = 2 * (self.label - self.a2)
        print("dL_da2: ", dL_da2)
        dL_dz2 = dL_da2 * (self.sigmoid(self.a2) * (1 - self.sigmoid(self.a2)))
        print("dL_dz2: ", dL_dz2)
        dL_dw2 = dL_da2 * dL_dz2 * self.a1
        print("dL_dw2: ", dL_dw2)
        
        dL_dw2 = dL_dw2.reshape(1, -1)
        
        print("dL_dw2 reshaped: ", dL_dw2.T)
        
        print("layer 2 weights: ", self.layer2.weights)
        
        self.layer2.weights -= self.learning_rate * dL_dw2.T
        
        dL_da1 = np.dot(self.layer2.weights.T, self.z2)
        dL_dz1 = dL_da1 * (self.sigmoid(self.a1) * (1 - self.sigmoid(self.a2)))
        dL_dw1 = np.dot(self.data.T, dL_dz1)
        
        self.layer1.weights -= self.learning_rate * dL_dw1
        
    
        


def main():
    hidden_layer_size = 5
    learning_rate = 0.001
    
    #Stochastic
    nn = NeuralNetwork(X.iloc[0], y[0], hidden_layer_size, learning_rate)

    result = nn.forward_pass()
    
    loss = nn.mse(result)
    
    back = nn.backward()

    print("NN after 1 forward pass: ", result)
    print("Loss after one pass: ", loss)
    
if __name__ == '__main__':
    main()