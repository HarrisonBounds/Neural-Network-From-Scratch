# Multi Class Cross Entropy

import torch.nn as nn
import torch
import pandas as pd


class NeuralNet(nn.Module):
    def __init__(self, input_size, hl_size, output_size):
        super(NeuralNet, self).__init__()
        self.k = hl_size
        self.linear1 = nn.Linear(input_size, hl_size)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(hl_size, output_size)
        self.softmax2 = nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.linear2(x)
        x = self.softmax2(x)
        return x

    def train_model(self, model, num_epochs, X_train, y_train, loss_func, optimizer):
        # Use Stochastic Gradient Descent (only use one data example at a time)
        for t in range(num_epochs):
            # print(f"Epoch {t+1}\n===================================================")
            for i in range(len(X_train)):

                x = X_train[i]
                y = y_train[i]

                y_pred = model(x)
                loss = loss_func(y_pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # print(f"Loss: {loss}")
    def validate_test(self, model, X_test, y_test, test_string):
        # Evaluating the model
        model.eval()  # Set the model to 'evaluation' mode

        # Disable gradient calculation for testing
        with torch.no_grad():
            correct = 0
            total = 0

            for i in range(len(X_test)):
                x = X_test[i]
                y = y_test[i]

                pred = model(x)[0]  # First probability in output.

                # Clipping for MSE
                if pred >= 0.5:
                    pred = 0
                elif pred < 0.5:
                    pred = 1

                if pred == y:
                    correct += 1

                total += 1

        accuracy = correct / total
        if test_string == "test":
            print(f"Accuracy on the test set: {accuracy} with k={self.k}")
        else:
            print(f"Accuracy on the valid set: {accuracy} with k={self.k}")
        return accuracy


def main():
    label = "label"

    # Alternatively, you could make a custom Dataloader class... didn't feel like it though
    # Format Data
    X_train = pd.read_csv("center_surround_train.csv")
    X_test = pd.read_csv("center_surround_test.csv")
    X_valid = pd.read_csv("center_surround_valid.csv")

    # Seperate labels
    y_train = X_train[label].values
    y_test = X_test[label].values
    y_valid = X_valid[label].values

    # Drop labels
    X_train = X_train.drop(label, axis=1).values
    X_test = X_test.drop(label, axis=1).values
    X_valid = X_valid.drop(label, axis=1).values

    # Convert to tensors for numpy to use
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.long)

    # Hyperparameters
    lr = 0.01
    num_epochs = 500
    hl_size = [2, 3, 5, 7, 9]
    outut_size = 2  # 2 for multi-class cross entropy, 1 for MSE
    acccuracies = {}
    for k in hl_size:
        # Build Network
        ff = NeuralNet(X_train.shape[1], k, outut_size)
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(ff.parameters(), lr=lr)

        ff.train_model(ff, num_epochs, X_train, y_train, loss_func, optimizer)
        valid_acc = ff.validate_test(ff, X_valid, y_valid, 'validate')
        test_acc = ff.validate_test(ff, X_test, y_test, 'test')
        acccuracies[k] = valid_acc, test_acc

    for k, v in acccuracies.items():
        print(
            f'Hidden Layer Size: {k}, ' +
            f'Validation Accuracy: {v[0]}, Test Accuracy: {v[1]}'
        )
    best_k = max(acccuracies, key=acccuracies.get)
    print(
        f'Best Hidden Layer Size: {best_k} with ' +
        f'Validation Accuracy: {acccuracies[best_k][0]} ' +
        f'and Test Accuracy: {acccuracies[best_k][1]}'
    )


if __name__ == '__main__':
    main()
