import torch.nn as nn
import torch
import pandas as pd
from torch import Tensor
from torch.optim import Optimizer
from typing import Callable, Tuple
from tqdm import tqdm

from nn_with_pytorch import NeuralNet
from plotter import plot_data, plot_decision_surface, plot_decision_regions


class HyperParams:
    def __init__(self, hidden_layer_size: int, learning_rate: float, loss_func: str):
        self._hl_size = hidden_layer_size
        self._lr = learning_rate
        self._output_size = 2 if loss_func == "MCE" else 1

    def __hash__(self):
        return hash((self.hl_size, self.lr, self.output_size))

    def __repr__(self):
        return f"[HLSize: {self.hl_size}, LR: {self.lr}, Loss: {'MCE' if self.output_size == 2 else 'MSE'}]"

    @property
    def hl_size(self):
        return self._hl_size

    @property
    def lr(self):
        return self._lr

    @property
    def output_size(self):
        return self._output_size


class NeuralNetEvaluator:
    def __init__(
            self,
            training_data: dict[str: str],
            test_data: dict[str: str],
            validation_data: dict[str: str],
            loss_functions: dict[str: Callable]):
        self.training_data = training_data
        self.test_data = test_data
        self.validation_data = validation_data
        self.loss_functions = loss_functions

        # Store the evaluated models as a dictionary that maps
        # a tuple of the dataset name and the hyperparams to a
        # dictionary of accuracy_names to accuracy values
        # Example Entry: ("dataset_name", HyperParams): {"valid_accuracy": 0.5, "test_accuracy": 0.6}
        self.evaluated_models: dict[
            Tuple[str, HyperParams]: dict[str: float]
        ] = {}

    def train_model(
            self,
            dataset_name: str,
            loss_func_name: str,
            hyperparams: HyperParams):
        # Assign hyperparams to default if not provided
        # Convert data to tensors
        X_train = pd.read_csv(self.training_data[dataset_name])
        X_test = pd.read_csv(self.test_data[dataset_name])
        X_validation = pd.read_csv(self.validation_data[dataset_name])
        # Seperate labels
        y_train = X_train["label"].values
        y_test = X_test["label"].values
        y_validation = X_validation["label"].values
        # Drop labels
        X_train = X_train.drop("label", axis=1).values
        X_test = X_test.drop("label", axis=1).values
        X_validation = X_validation.drop("label", axis=1).values
        # Convert to tensors for numpy to use
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        X_validation = torch.tensor(X_validation, dtype=torch.float32)
        # For MCE, we need to convert the labels to long
        if loss_func_name == "MCE":
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_test = torch.tensor(y_test, dtype=torch.long)
            y_validation = torch.tensor(y_validation, dtype=torch.long)
        elif loss_func_name == "MSE":
            y_train = torch.tensor(y_train, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)
            y_validation = torch.tensor(y_validation, dtype=torch.float32)
        # Build network with loss function and optimizer
        neural_network = NeuralNet(
            X_train.shape[1],
            hl_size=hyperparams.hl_size,
            output_size=2 if loss_func_name == "MCE" else 1
        )
        loss_func = self.loss_functions[loss_func_name]
        optimizer = torch.optim.Adam(
            neural_network.parameters(),
            lr=hyperparams.lr
        )
        # Train the model
        neural_network.train_model(
            neural_network,
            num_epochs=500,
            X_train=X_train,
            y_train=y_train,
            loss_func=loss_func,
            optimizer=optimizer
        )
        # Evaluate the model
        valid_accuracy = neural_network.validate_test(
            neural_network,
            X_validation,
            y_validation,
            loss_func_name
        )
        test_accuracy = neural_network.validate_test(
            neural_network,
            X_test,
            y_test,
            loss_func_name
        )
        # Save the accuracy to the evaluated models
        key = (dataset_name, hyperparams)
        self.evaluated_models[key] = {
            "valid_accuracy": valid_accuracy,
            "test_accuracy": test_accuracy
        }
        print(
            f"Finished training {dataset_name} with Hyperparams: {hyperparams}" +
            f" with loss function: {loss_func_name}"
        )

    def print_evaluated_models(self):
        unique_datasets = set([key[0] for key in self.evaluated_models.keys()])
        print(
            f'Evaluated {len(self.evaluated_models)} models over ' +
            f'{len(unique_datasets)} datasets'
        )
        for key in self.evaluated_models.keys():
            dataset_name, hyper_params = key
            valid_accuracy = self.evaluated_models[key]["valid_accuracy"]
            test_accuracy = self.evaluated_models[key]["test_accuracy"]
            print("======================================")
            print(f"{dataset_name} with Hyperparams: {hyper_params}")
            print(f"Validation Accuracy: {valid_accuracy}")
            print(f"Test Accuracy: {test_accuracy}")
            print("======================================\n")

    def find_best_hyperparams_for_dataset(self, dataset_name: str):
        # This assumes that the models are already trained and stored
        best_test_accuracy = 0
        best_valid_accuracy = 0
        best_hyperparams = None
        # dataset_models should be all the models where the dataset_name is the key
        dataset_models = [
            key for key in self.evaluated_models.keys() if key[0] == dataset_name
        ]
        for key in dataset_models:
            dataset, hp = key
            valid_accuracy = self.evaluated_models[key]["valid_accuracy"]
            test_accuracy = self.evaluated_models[key]["test_accuracy"]
            if valid_accuracy > best_valid_accuracy:
                best_test_accuracy = test_accuracy
                best_valid_accuracy = valid_accuracy
                best_hyperparams = hp
        return best_hyperparams, best_valid_accuracy, best_test_accuracy

    def plot_learning_curve(self, dataset_name: str):
        # Plot the learning curve for training and validation loss as
        # a function of training epochs
        pass

    def plot_learned_decision_surface(self, dataset_name: str):
        # Plot the learned decision surface along with observations from the test set
        pass


def main():
    print(f"WARNING: This script will take a long time to run ...")
    evaluator = NeuralNetEvaluator(
        training_data={"xor": "xor_train.csv",
                       "center_surround": "center_surround_train.csv",
                       "spiral": "spiral_train.csv",
                       "two_gaussians": "two_gaussians_train.csv"
                       },
        test_data={"xor": "xor_test.csv",
                   "center_surround": "center_surround_test.csv",
                   "spiral": "spiral_test.csv",
                   "two_gaussians": "two_gaussians_test.csv"
                   },
        validation_data={"xor": "xor_valid.csv",
                         "center_surround": "center_surround_valid.csv",
                         "spiral": "spiral_valid.csv",
                         "two_gaussians": "two_gaussians_valid.csv"
                         },
        loss_functions={"MCE": nn.CrossEntropyLoss(), "MSE": nn.MSELoss()}
    )
    datasets = ["xor", "center_surround", "spiral", "two_gaussians"]
    hidden_layer_sizes = [2, 3, 5, 7, 9]
    loss = "MSE"
    for dataset in tqdm(datasets, desc="Datasets"):
        for hl_size in hidden_layer_sizes:
            hp = HyperParams(hl_size, 0.01, loss)
            evaluator.train_model(dataset, loss, hp)
    evaluator.print_evaluated_models()
    for dataset in datasets:
        best_hp, valid_acc, test_acc = evaluator.find_best_hyperparams_for_dataset(
            dataset
        )
        print("======================================")
        print(f"Best Hyperparams for {dataset}: {best_hp}")
        print(f"Validation Accuracy: {valid_acc}")
        print(f"Test Accuracy: {test_acc}")
        print("======================================\n")


if __name__ == '__main__':
    main()
