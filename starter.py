import pandas as pd
import numpy as np

train_filename = "center_surround_train.csv"
validate_filename = "center_surround_valid.csv"
test_filename = "center_surround_test.csv"
label_string = "label"

X = pd.read_csv(train_filename)
V = pd.read_csv(validate_filename)
T = pd.read_csv(test_filename)

X = X.drop(label_string, axis=1)

print("X shape: ", X.shape)
print("V shape: ", V.shape)
print("T shape: ", T.shape)

class Node():
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros(output_size)
        
    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases
        
        
class NeuralNetwork():
    def __init__(self, input_data, hl_size):
        self.hl_size = hl_size
        self.data = input_data
        self.layer1 = Node(len(input_data), self.hl_size)
        self.layer2 = Node(self.hl_size, 2) #Two for binary classification
        
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def forward_pass(self):
        z1 = self.layer1.forward(self.data.T)
        a1 = self.sigmoid(z1)
        z2 = self.layer2.forward(a1)
        a2 = self.sigmoid(z2)
        return a2

hidden_layer_size = 5
nn = NeuralNetwork(X, hidden_layer_size)

result = nn.forward_pass()

print("NN after 1 forward pass: ", result)      