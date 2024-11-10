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

class Node():
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros(output_size)
        
    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases
        
        
class NeuralNetwork():
    def __init__(self, input_data, labels, hl_size):
        self.hl_size = hl_size
        self.data = input_data
        self.layer1 = Node(len(input_data), self.hl_size)
        self.layer2 = Node(self.hl_size, 1) #One for binary classification
        self.labels = labels
        
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        pass
    
    def mse(self, y_pred):
        return np.sum((self.labels - y_pred)**2) #Dont need 1/N because we are using stochastic gradient descent (processing one example at a time), so N would be 1
                            
    def forward_pass(self):
        z1 = self.layer1.forward(self.data.T)
        a1 = self.sigmoid(z1)
        z2 = self.layer2.forward(a1)
        a2 = self.sigmoid(z2)
        return a2


def main():
    hidden_layer_size = 5
    
    #Stochastic
    nn = NeuralNetwork(X.iloc[0], y, hidden_layer_size)

    result = nn.forward_pass()
    
    loss = nn.mse(result)

    print("NN after 1 forward pass: ", result)
    print("Loss after one pass: ", loss)
    
if __name__ == '__main__':
    main()