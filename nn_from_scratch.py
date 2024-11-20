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
    
    def backward(self, y_pred, X, y_true):
        dL_da2 = y_pred - y_true
        da2_dz2 = y_pred * (1 - y_pred) #sigmoid derivative
        dL_dz2 = dL_da2 * da2_dz2
        dz2_dw2 =  self.a1
        
        dL_dw2 = np.dot(dz2_dw2.T, dL_dz2)
        dL_b2 = np.sum(dL_dz2, axis=0, keepdims=True)

        
        dL_da1 = np.dot(dL_dz2, self.layer2.weights.T)
        dL_dz1 = dL_da1 * (self.a1 * (1 - self.a1))
        
        X = X.reshape(-1, 1)
        dL_dw1 = np.dot(X, dL_dz1)
        dL_b1 = np.sum(dL_dz1, axis=0, keepdims=True)

    
        self.layer1.weights -= self.learning_rate * dL_dw1
        self.layer1.biases -= self.learning_rate * dL_b1
        self.layer2.weights -= self.learning_rate * dL_dw2
        self.layer2.biases -= self.learning_rate * dL_b2
        
    def train(self, num_epochs, X_train, y_train, nn):
        #Stochastic Gradient descent
        for i in range(num_epochs):
            print(f"Epoch {i}==================================================")
            for j in range(len(X_train)):  
                input_data = X_train[j]
                label = y_train[j]
                
                nn.data = input_data
                nn.label = label

                y_pred = nn.forward_pass()
                loss = nn.mse(y_train[j], y_pred)
                nn.backward(y_pred, input_data, y_train[j])
                
            print(f"Loss: {loss}")
            
    def evaluate(self, X, y, nn):
        correct = 0
        total = 0
        for j in range(len(X)):
            input_data = X[j]
            nn.data = input_data
            
            y_pred = nn.forward_pass()
            
            if y_pred >= 0.5:
                y_pred = 1
            elif y_pred < 0.5:
                y_pred = 0
                
            if y_pred == y[j]:
                correct += 1
                
            total += 1
            
        accuracy = correct / total
        return accuracy
    
def main():
    hidden_layer_size = 9
    learning_rate = 0.01
    num_epochs = 700
    np.random.seed(11)
    label_string = 'label'
    valid_accuracy_list = []
    test_accuracy_list = []
    
    train_list = ["center_surround_train.csv", "spiral_train.csv",
                  "two_gaussians_train.csv", "xor_train.csv"]
    test_list = ["center_surround_test.csv", "spiral_test.csv",
                 "two_gaussians_test.csv", "xor_test.csv"]
    valid_list = ["center_surround_valid.csv", "spiral_valid.csv",
                  "two_gaussians_valid.csv", "xor_valid.csv"]

    for i in range(len(train_list)):
        X_train = pd.read_csv(train_list[i])
        X_test = pd.read_csv(test_list[i])
        X_valid = pd.read_csv(valid_list[i])

        # Seperate labels
        y_train = X_train[label_string].values
        y_test = X_test[label_string].values
        y_valid = X_valid[label_string].values

        # Drop labels
        X_train = X_train.drop(label_string, axis=1).values
        X_test = X_test.drop(label_string, axis=1).values
        X_valid = X_valid.drop(label_string, axis=1).values 

    
        nn = NeuralNetwork(X_train[0], hidden_layer_size, learning_rate)
        
        nn.train(num_epochs, X_train, y_train, nn)
        valid_accuracy = nn.evaluate(X_valid, y_valid, nn)
        test_accuracy = nn.evaluate(X_valid, y_valid, nn)
        
        valid_accuracy_list.append(valid_accuracy)
        test_accuracy_list.append(test_accuracy)
        
    for i in range(len(test_list)):
        print(f"Validation Accuracy for dataset {valid_list[i]}: {valid_accuracy_list[i]}")
        print(f"Test Accuracy for dataset {test_list[i]}: {test_accuracy_list[i]}\n")
        
    
if __name__ == '__main__':
    main()