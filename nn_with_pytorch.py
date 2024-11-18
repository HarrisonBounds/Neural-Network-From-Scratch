"""Binary Classification in pytorch using either multi-class cross entropy or mean squared error loss functions"""

import torch.nn as nn
import torch
import pandas as pd
from torch import Tensor
from torch.optim import Optimizer
from typing import Callable


class NeuralNet(nn.Module):
    def __init__(self, input_size: int, hl_size: int, output_size: int):
        """Initialize the network with the size of each layer

        Args:
            input_size (int): Number of features in the dataset
            hl_size (int): hidden layer size
            output_size (int): output size (1 for MSE loss, 2 for MCE loss)
        """
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hl_size)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(hl_size, output_size)
        if output_size == 2:
            self.output = nn.Softmax()
        elif output_size == 1:
            self.output = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the neural network

        Args:
            x (Tensor): Input data in the form of a tensor

        Returns:
            Tensor: data after the nework has processed it (output tensor)
        """
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.linear2(x)
        x = self.output(x)
        return x

    def train_model(self, model: nn.Module, num_epochs: int, X_train: Tensor, y_train: Tensor, loss_func: Callable, optimizer: Optimizer):
        """Use Stochastic Gradient Descent to train one example at a time

        Args:
            model (nn.Module): Model created by the forward pass
            num_epochs (int): Number of times to train the entire dataset
            X_train (Tensor): Training data without labels
            y_train (Tensor): Labels for the training data
            loss_func (Callable): callable function to perform the loss calculation on the output
            optimizer (Optimizer): Built in optimzer to perform momentum based learning
        """
        learning_curve = []
        for epoch in range(num_epochs):
            epoch_loss = 0
            for i in range(len(X_train)):

                x = X_train[i]
                y = y_train[i]

                y_pred = model(x)
                loss = loss_func(y_pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate loss for the epoch
                epoch_loss += loss.item()
            # Learning curve should be average loss per epoch
            learning_curve.append(epoch_loss / len(X_train))
        return learning_curve

    def validate_test(self, model: nn.Module, X_test: Tensor, y_test: Tensor, loss_func_label: str) -> float:
        """Test the validation and test set accuracies by divinding the correct predictions by the total predicitions

        Args:
            model (nn.Module): Model created by the forward pass
            X_test (Tensor): Testing/Validation data without labels
            y_test (Tensor): Labels for the testing/validation data
            loss_func_label (str): Specifies either MSE or MCE loss

        Returns:
            float: accuracy evaluation metric to see how well the network is performing
        """
        model.eval()  # Set the model to 'evaluation' mode

        # Disable gradient calculation for testing
        with torch.no_grad():
            correct = 0
            total = 0

            for i in range(len(X_test)):
                x = X_test[i]
                y = y_test[i]

                if loss_func_label == "MCE":
                    # Second probability in output: (Second probability so we can keep the logic the same for clipping)
                    pred = model(x)[1]
                elif loss_func_label == "MSE":
                    pred = model(x)

                # Clipping: If the second element has higher than a 50% probability, then it is of class 1, otherwise class 0
                if pred >= 0.5:
                    pred = 1
                elif pred < 0.5:
                    pred = 0

                if pred == y:
                    correct += 1

                total += 1

        accuracy = correct / total

        return accuracy


def main():
    # Hyperparameters
    lr = 0.01
    num_epochs = 500
    hl_size = 5
    outut_size = 2  # 2 for multi-class cross entropy, 1 for MSE
    label = "label"
    loss_function_label = "MSE"  # Either MCE or MSE
    valid_accuracy_list = []
    test_accuracy_list = []

    train_list = ["center_surround_train.csv", "spiral_train.csv",
                  "two_gaussians_train.csv", "xor_train.csv"]
    test_list = ["center_surround_test.csv", "spiral_test.csv",
                 "two_gaussians_test.csv", "xor_test.csv"]
    valid_list = ["center_surround_valid.csv", "spiral_valid.csv",
                  "two_gaussians_valid.csv", "xor_valid.csv"]

    # Alternatively, you could make a custom Dataloader class... didn't feel like it though
    for i in range(len(train_list)):
        # Format Data
        print(
            f"Dataset {i+1}: {train_list[i]} using {loss_function_label} loss function\n")

        X_train = pd.read_csv(train_list[i])
        X_test = pd.read_csv(test_list[i])
        X_valid = pd.read_csv(valid_list[i])

        # Seperate labels
        y_train = X_train[label].values
        y_test = X_test[label].values
        y_valid = X_valid[label].values

        # Drop labels
        X_train = X_train.drop(label, axis=1).values
        X_test = X_test.drop(label, axis=1).values
        X_valid = X_valid.drop(label, axis=1).values

        # Convert to tensors for numpy to use
        if loss_function_label == "MCE":
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.long)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.long)
            X_valid = torch.tensor(X_valid, dtype=torch.float32)
            y_valid = torch.tensor(y_valid, dtype=torch.long)
        elif loss_function_label == "MSE":
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)
            X_valid = torch.tensor(X_valid, dtype=torch.float32)
            y_valid = torch.tensor(y_valid, dtype=torch.float32)

        # Build network
        if loss_function_label == "MCE":
            ff = NeuralNet(X_train.shape[1], hl_size, 2)
            loss_func = nn.CrossEntropyLoss()
        elif loss_function_label == "MSE":
            ff = NeuralNet(X_train.shape[1], hl_size, 1)
            loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(ff.parameters(), lr=lr)

        # Train Model
        ff.train_model(ff, num_epochs, X_train, y_train, loss_func, optimizer)

        # Evaluate Model
        valid_accuracy = ff.validate_test(
            ff, X_valid, y_valid, loss_function_label)
        test_accuracy = ff.validate_test(
            ff, X_test, y_test, loss_function_label)

        # Append evaluation metrics to a list for easy access
        valid_accuracy_list.append(valid_accuracy)
        test_accuracy_list.append(test_accuracy)

    # Print evaluation Metrics for each dataset
    for i in range(len(valid_accuracy_list)):
        print(f"{train_list[i]}")
        print("===================")
        print(f"Validation Accuracy: {valid_accuracy_list[i]}")
        print(f"Test Accuracy: {test_accuracy_list[i]}")


if __name__ == '__main__':
    main()
