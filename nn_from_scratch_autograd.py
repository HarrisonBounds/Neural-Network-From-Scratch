import pandas as pd
import numpy as np
from autograd import grad, jacobian
import autograd.numpy as anp


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

        gradients = []
        w = self.weights[-1]
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

    def update_weights(self, gradients):
        for i in range(len(gradients)):
            self.weights[i] = self.weights[i] - self.learning_rate*gradients[i]
        self.params = [self.weights[:-1], self.weights[-1]]
        return self.params

    def train(self, num_epochs, X_train, y_train):
        # Stochastic Gradient descent
        for i in range(num_epochs):
            for j in range(len(X_train)):
                input_data = X_train[j]
                label = y_train[j]

                p_true = self.model(input_data, self.params)
                gradients = self.back_propagation(input_data, label)
                _ = self.update_weights(gradients)

    def evaluate(self, X, y):
        correct = 0
        total = 0
        for j in range(len(X)):
            input_data = X[j]
            y_pred = 0
            p = self.model(input_data, self.params)

            if p[0] >= 0.5:
                y_pred = 1
            elif p[0] < 0.5:
                y_pred = 0

            if y_pred == y[j]:
                correct += 1

            total += 1

        accuracy = correct / total
        return accuracy


def main():
    learning_rate = 0.01
    num_epochs = 200
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

        print(X_train)

        nn = NN([2, 12, 2], learning_rate)

        nn.train(num_epochs, X_train, y_train)
        valid_accuracy = nn.evaluate(X_valid, y_valid)
        test_accuracy = nn.evaluate(X_test, y_test)

        valid_accuracy_list.append(valid_accuracy)
        test_accuracy_list.append(test_accuracy)

    for i in range(len(test_list)):
        print(f"Validation Accuracy for dataset {
              valid_list[i]}: {valid_accuracy_list[i]}")
        print(f"Test Accuracy for dataset {
              test_list[i]}: {test_accuracy_list[i]}\n")


if __name__ == '__main__':
    main()
