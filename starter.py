import pandas as pd
import numpy as np
from autograd import grad, jacobian
import autograd.numpy as anp


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
        self.layer2 = Node(self.hl_size, 1)  # One for binary classification
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
        dz2_dw2 = self.a1

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


class NN():
    def __init__(self, layers, learning_rate, weights=0):
        self.weights = []
        self.learning_rate = learning_rate
        self.num_hidden_layers = len(layers) - 2
        self.x = [0]*layers[0]
        self.q = [[0]*layers[i+1]
                  for i in range(self.num_hidden_layers)]
        self.h = [[0]*layers[i+1]
                  for i in range(self.num_hidden_layers)]
        self.z = [0]*layers[-1]
        self.p = [0]*layers[-1]

        if weights == 0:
            for i in range(len(layers) - 1):
                input = layers[i]
                output = layers[i+1]

                weight = anp.random.randn(input+1, output)
                self.weights.append(weight)
        else:
            self.weights = weights

        self.params = [self.weights[:-1], self.weights[-1]]

    def sigmoid(self, x):
        return 1/(1+anp.exp(-x))

    def softmax(self, z):
        return anp.exp(z)/anp.sum(anp.exp(z))

    def loss(self, p, y):
        return -anp.sum(y*anp.log(p) + (anp.ones_like(y)-y)*anp.log(anp.ones_like(p)-p))

    def linear_transform(self, x, w):
        return w[0] + anp.dot(x.T, w[1:])

    def ft(self, x, w):
        for i in range(len(w)):
            x = self.linear_transform(x, w[i])
            self.q[i] = x

            x = self.sigmoid(x).T
            self.h[i] = x

        return x

    def model(self, x, params):
        self.x = x
        f = self.ft(x, params[0])

        z = self.linear_transform(f, params[1])
        self.z = z

        p = self.softmax(z)
        self.p = p

        return p.T

    def back_propagation(self, x, y):
        grad_L = grad(self.loss)
        grad_sig = jacobian(self.sigmoid)
        grad_soft = jacobian(self.softmax)

        dLdp = grad_L(self.p, y)
        dpdz = grad_soft(self.z).diagonal()
        dzdw = self.h[0]
        dhdq = grad_sig(anp.array(self.q[0])).diagonal()
        dqdw = self.x
        dzdh = self.weights[-1]
        print(dzdw)
        print(dpdz)
        print(dLdp)
        print(dhdq)

        gradients = []
        w = self.weights[-1]
        print(len(w), len(w[0]))
        grads = np.empty((len(w), len(w[0])))
        for i in range(len(w)):
            for j in range(len(w[0])):
                if i == 0:
                    grads[i][j] = dLdp[j]*dpdz[j]*1
                else:
                    grads[i][j] = dLdp[j]*dpdz[j]*dzdw[i-1]

        gradients.append(grads)

        w = self.weights[0]
        grads = np.empty((len(w), len(w[0])))
        for i in range(len(w)):
            for j in range(len(w[0])):
                if i == 0:
                    grads[i][j] = np.sum(dhdq[j]*1*(dzdh[j+1]*dpdz*dLdp))
                else:
                    grads[i][j] = np.sum(
                        dhdq[j]*dqdw[i-1]*(dzdh[j+1]*dpdz*dLdp))

        gradients.append(grads)
        gradients.reverse()

        return gradients


def main():
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


if __name__ == '__main__':
    main()
