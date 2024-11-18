import torch.nn as nn
import torch
import pandas as pd
from torch import Tensor
from torch.optim import Optimizer
from typing import Callable, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        self.used_regularizer = ""

        # Store the evaluated models as a dictionary that maps
        # a tuple of the dataset name and the hyperparams to a
        # dictionary of accuracy_names to accuracy values
        # Example Entry: ("dataset_name", HyperParams): {"valid_accuracy": 0.5, "test_accuracy": 0.6}
        self.evaluated_models: dict[
            Tuple[str, HyperParams]: dict[str: float]
        ] = {}
        # Store the best models as a map of
        # dataset_name_loss to the model itself
        self.all_models: dict[str: dict[HyperParams: NeuralNet]] = {}
        self.best_models: dict[str: NeuralNet] = {}

    def train_model(
            self,
            dataset_name: str,
            loss_func_name: str,
            regularizer: str,
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
        if regularizer is not None or regularizer != "":
            self.used_regularizer = regularizer
        neural_network = NeuralNet(
            X_train.shape[1],
            hl_size=hyperparams.hl_size,
            output_size=2 if loss_func_name == "MCE" else 1,
            regularizer=regularizer if regularizer != "" else None
        )
        loss_func = self.loss_functions[loss_func_name]
        optimizer = torch.optim.Adam(
            neural_network.parameters(),
            lr=hyperparams.lr
        )
        # Train the model
        training_losses, validation_losses = neural_network.train_model(
            neural_network,
            num_epochs=500,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_validation,
            y_valid=y_validation,
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
            "test_accuracy": test_accuracy,
            "training_losses": training_losses,
            "validation_losses": validation_losses
        }
        dataset_loss = f"{dataset_name}_{loss_func_name}"
        if dataset_loss not in self.all_models:
            self.all_models[dataset_loss] = {}
        self.all_models[dataset_loss][hyperparams] = neural_network
        print(
            f"Finished training {dataset_name} with Hyperparams: {hyperparams}" +
            f" with loss function: {
                loss_func_name} and regularizer: {"None" if regularizer == "" else regularizer}"
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

    def find_best_hyperparams_for_dataset(self, dataset_name: str, loss_func_name: str):
        # This assumes that the models are already trained and stored
        best_test_accuracy = 0
        best_valid_accuracy = 0
        best_hyperparams = None
        # dataset_models should be all the models where the dataset_name is the key
        loss_output_size = 2 if loss_func_name == "MCE" else 1
        dataset_models = [
            key for key in self.evaluated_models.keys() if key[0] == dataset_name and key[1].output_size == loss_output_size
        ]
        for key in dataset_models:
            dataset, hp = key
            valid_accuracy = self.evaluated_models[key]["valid_accuracy"]
            test_accuracy = self.evaluated_models[key]["test_accuracy"]
            if valid_accuracy > best_valid_accuracy:
                best_test_accuracy = test_accuracy
                best_valid_accuracy = valid_accuracy
                best_hyperparams = hp
        dataset_loss = f"{dataset_name}_{loss_func_name}"
        model_dict = self.all_models[dataset_loss]
        # Find the entry that matches the best hyperparams
        best_model = model_dict[best_hyperparams]
        self.best_models[dataset_loss] = best_model
        return best_hyperparams, best_valid_accuracy, best_test_accuracy

    def plot_learning_curves(self, dataset_name: str, best_hp: HyperParams):
        # Plot the learning curve for training and validation loss as
        # a function of training epochs
        # find the epoch_losses for the best hyperparams
        key = (dataset_name, best_hp)
        loss = "MCE" if best_hp.output_size == 2 else "MSE"
        training_losses = self.evaluated_models[key]["training_losses"]
        validation_losses = self.evaluated_models[key]["validation_losses"]
        plt.figure()
        if self.used_regularizer is not None or self.used_regularizer != "":
            reg_tag = f"_{self.used_regularizer}"
        plt.plot(
            range(len(training_losses)), training_losses,
            label=f"{dataset_name}_{loss}_{
                best_hp.hl_size}{reg_tag}_training_loss"
        )
        plt.plot(
            range(len(validation_losses)), validation_losses,
            label=f"{dataset_name}_{loss}_{
                best_hp.hl_size}{reg_tag}_validation_loss"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(
            f"{dataset_name} (k={best_hp.hl_size}, " +
            f"reg={'None' if self.used_regularizer == '' else self.used_regularizer}" +
            f", loss={loss}) Learning Curve"
        )
        plt.legend(["Training Loss", "Validation Loss"])
        plt.savefig(
            f"plots/{dataset_name}_{best_hp.hl_size}_{
                loss}{reg_tag}_learning_curve.png"
        )

    def plot_learned_decision_surfaces(self, dataset_name: str, loss_func_name: str):
        # Plot the learned decision surface along with observations from the test set
        dataset_loss = f"{dataset_name}_{loss_func_name}"
        # plot_decision_surface(model=self.best_models[dataset_loss])
        X_test = pd.read_csv(self.test_data[dataset_name])
        y_test = X_test["label"].values
        plot_decision_regions(
            features=X_test.drop("label", axis=1).values,
            targets=y_test,
            model=self.best_models[dataset_loss]
        )


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
    losses = ["MCE", "MSE"]
    regularizers = ["", "norm", "orthogonal"]
    # After running and manually inspecting the results,
    # these are the best HPs for each dataset and loss function
    # These should also be used when using regularizers
    best_hps_map = {
        "xor_MCE": HyperParams(7, 0.01, "MCE"),
        "xor_MSE": HyperParams(9, 0.01, "MSE"),
        "center_surround_MCE": HyperParams(3, 0.01, "MCE"),
        "center_surround_MSE": HyperParams(3, 0.01, "MSE"),
        "spiral_MCE": HyperParams(9, 0.01, "MCE"),
        "spiral_MSE": HyperParams(9, 0.01, "MSE"),
        "two_gaussians_MCE": HyperParams(2, 0.01, "MCE"),
        "two_gaussians_MSE": HyperParams(3, 0.01, "MSE")
    }
    for reg in regularizers:
        for dataset in tqdm(datasets, desc="Datasets"):
            for loss in losses:
                dataset_loss = f"{dataset}_{loss}"
                hp = best_hps_map[dataset_loss]
                evaluator.train_model(dataset, loss, reg, hp)
        # Uncomment this block to train models with different hyperparams
        # for dataset in tqdm(datasets, desc="Datasets"):
        #     for hl_size in hidden_layer_sizes:
        #         for loss in losses:
        #             hp = HyperParams(hl_size, 0.01, loss)
        #             evaluator.train_model(dataset, loss, hp)
        evaluator.print_evaluated_models()
        for dataset in datasets:
            for loss_name in losses:
                best_hp, valid_acc, test_acc = evaluator.find_best_hyperparams_for_dataset(
                    dataset, loss_name
                )
                print("======================================")
                print(f"Best Hyperparams for {dataset}: {best_hp}")
                print(f"Validation Accuracy: {valid_acc}")
                print(f"Test Accuracy: {test_acc}")
                print("======================================\n")
                evaluator.plot_learning_curves(dataset, best_hp)
                # evaluator.plot_learned_decision_surfaces(dataset, loss_name)


if __name__ == '__main__':
    main()
