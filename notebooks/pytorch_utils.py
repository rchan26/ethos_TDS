import os
import random
from typing import Union, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import TensorDataset
from tqdm import tqdm


class KFolds:
    """
    Class to split the data into different folds
    """

    def __init__(
        self,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: int = 42,
    ):
        """
        Class to split the data into different folds

        Parameters
        ----------
        x_data : torch.Tensor
            Features for prediction
        y_data : torch.Tensor
            Variable to predict
        n_splits : int, optional
            Number of splits / folds, by default 5
        shuffle : bool, optional
            Whether or not to shuffle the dataset, by default False
        random_state : int, optional
            Seed number, by default 42

        Raises
        ------
        ValueError
            if `n_splits` < 2
        ValueError
            if `x_data` and `y_data` do not have the same number of records
            (number of rows in `x_data` should equal the length of `y_data`)
        """
        if n_splits < 2:
            raise ValueError("n_splits should be at least 2")
        if x_data.shape[0] != y_data.shape[0]:
            raise ValueError(
                "x_data and y_data do not have compatible shapes "
                + "(need to have same number of samples)"
            )
        self.x_data = x_data
        self.y_data = y_data
        self.n_splits = n_splits
        self.shuffle = shuffle
        if self.shuffle:
            self.random_state = random_state
        else:
            self.random_state = None
        self.fold = KFold(n_splits=self.n_splits,
                          shuffle=self.shuffle,
                          random_state=self.random_state)
        self.fold_indices = list(self.fold.split(X=x_data))

    def get_splits(
        self,
        fold_index: int,
        dev_size: float = 0.33,
        as_DataLoader=False,
        data_loader_args: dict = {"batch_size": 1, "shuffle": True},
    ) -> Union[
        Tuple[DataLoader, DataLoader, DataLoader],
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        """
        Obtains the data from a particular fold

        Parameters
        ----------
        fold_index : int
            Which fold to obtain data for
        dev_size : float, optional
            Proportion of training data to use as validation data, by default 0.33
        as_DataLoader : bool, optional
            Whether or not to return as `torch.utils.data.dataloader.DataLoader` objects
            ready to be passed into PyTorch model, by default False
        data_loader_args : _type_, optional
            Any keywords to be passed in obtaining the
            `torch.utils.data.dataloader.DataLoader` object,
            by default {"batch_size": 1, "shuffle": True}

        Returns
        -------
        - If `as_DataLoader` is True, return tuple of
        `torch.utils.data.dataloader.DataLoader` objects:
          - First element is training dataset
          - Second element is validation dataset
          - Third element is testing dataset
        - If `as_DataLoader` is False, returns tuple of `torch.Tensors`:
          - First element is features for testing dataset
          - Second element is labels for testing dataset
          - First element is features for validation dataset
          - Second element is labels for validation dataset
          - First element is features for training dataset
          - Second element is labels for training dataset

        Raises
        ------
        ValueError
            if the requested fold_index is not valid
        """
        if fold_index not in list(range(self.n_splits)):
            raise ValueError(
                f"There are {self.n_splits} folds, so "
                + f"fold_index must be in {list(range(self.n_splits))}"
            )
        # obtain train and test indices for provided fold_index
        train_index = self.fold_indices[fold_index][0]
        test_index = self.fold_indices[fold_index][1]
        # obtain a validation set from the training set
        train_index, valid_index = train_test_split(
            train_index,
            test_size=dev_size,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

        x_train = self.x_data[train_index]
        y_train = self.y_data[train_index]
        x_valid = self.x_data[valid_index]
        y_valid = self.y_data[valid_index]
        x_test = self.x_data[test_index]
        y_test = self.y_data[test_index]
        
        print(y_train.type())

        if as_DataLoader:
            train = TensorDataset(x_train, y_train)
            valid = TensorDataset(x_valid, y_valid)
            test = TensorDataset(x_test, y_test)

            train_loader = DataLoader(dataset=train, **data_loader_args)
            valid_loader = DataLoader(dataset=valid, **data_loader_args)
            test_loader = DataLoader(dataset=test, **data_loader_args)

            return train_loader, valid_loader, test_loader
        else:
            return x_test, y_test, x_valid, y_valid, x_train, y_train


def set_seed(seed: int) -> None:
    """
    Helper function for reproducible behavior to set the seed in
    `random`, `numpy`, `torch` (if installed).

    Parameters
    ----------
    seed : int
        Seed number
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def validation_pytorch(
    model: nn.Module,
    valid_loader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    verbose: bool = False,
    verbose_epoch: int = 100,
) -> Tuple[float, float, float]:
    """
    Evaluates the PyTorch model to a validation set and
    returns the total loss, accuracy and F1 score

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model which inherits from the `torch.nn.Module` class
    valid_loader : DataLoader
        Validation dataset as `torch.utils.data.dataloader.DataLoader` object
    criterion : torch.nn.Module
        Loss function which inherits from the `torch.nn.Module` class
    epoch : int
        Epoch number
    verbose : bool, optional
        Whether or not to print progress, by default False
    verbose_epoch : int, optional
        How often to print progress during the epochs, by default 100

    Returns
    -------
    Tuple[float, float, float]
        Current average loss, accuracy and F1 score
    """
    # sets the model to evaluation mode
    model.eval()
    number_of_labels = 0
    total_loss = 0
    labels = torch.empty((0))
    predicted = torch.empty((0))
    with torch.no_grad():
        for emb_v, labels_v in valid_loader:
            # make prediction
            outputs = model(emb_v)
            _, predicted_v = torch.max(outputs.data, 1)
            number_of_labels += labels_v.size(0)
            # compute loss
            loss_v = criterion(outputs, labels_v)
            total_loss += loss_v.item()
            # save predictions and labels
            labels = torch.cat([labels, labels_v])
            predicted = torch.cat([predicted, predicted_v])
        # compute accuracy and f1 score
        accuracy = ((predicted == labels).sum() / number_of_labels).item()
        f1_v = metrics.f1_score(labels, predicted, average="macro")
        if verbose:
            if epoch % verbose_epoch == 0:
                print(
                    f"Epoch: {epoch+1} || "
                    + f"Loss: {total_loss / len(valid_loader)} || "
                    + f"Accuracy: {accuracy} || "
                    + f"F1-score: {f1_v}."
                )

        return total_loss / len(valid_loader), accuracy, f1_v

def training_pytorch(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    num_epochs: int,
    seed: Optional[int] = 42,
    patience: Optional[int] = 3,
    verbose: bool = False,
    verbose_epoch: int = 100,
    verbose_item: int = 1000,
) -> nn.Module:
    """
    Trains the PyTorch model using some training dataset and
    uses a validation dataset to determine if early stopping is used

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model which inherits from the `torch.nn.Module` class
    train_loader : torch.utils.data.dataloader.DataLoader
        Training dataset as `torch.utils.data.dataloader.DataLoader` object
    valid_loader : torch.utils.data.dataloader.DataLoader
        Validation dataset as `torch.utils.data.dataloader.DataLoader` object
    criterion : torch.nn.Module
        Loss function which inherits from the `torch.nn.Module` class
    optimizer : torch.optim.optimizer.Optimizer
        PyTorch Optimizer
    num_epochs : int
        Number of epochs
    seed : Optional[int], optional
        Seed number, by default 42
    patience : Optional[int], optional
        Patience parameter for early stopping rule, by default 3
    verbose : bool, optional
        Whether or not to print progress, by default False
    verbose_epoch : int, optional
        How often to print progress during the epochs, by default 100
    verbose_item : int, optional
        How often to print progress when iterating over items
        in training set, by default 1000

    Returns
    -------
    torch.nn.Module
        Trained PyTorch model
    """
    # sets the model to training mode
    model.train()
    set_seed(seed)
    # early stopping parameters
    last_metric = 0
    trigger_times = 0
    # model train & validation per epoch
    for epoch in tqdm(range(num_epochs)):
        for i, (emb, labels) in enumerate(train_loader):
            # perform training by performing forward and backward passes
            optimizer.zero_grad()
            outputs = model(emb)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # show training progress
            if verbose:
                if (epoch % verbose_epoch == 0) and (i % verbose_item == 0):
                    print(
                        f"Epoch: {epoch+1}/{num_epochs} || "
                        + f"Item: {i}/{len(train_loader)} || "
                        + f"Loss: {loss.item()}"
                    )
        # show training progress
        if verbose:
            if epoch % verbose_epoch == 0:
                print("-" * 50)
                print(
                    f"##### Epoch: {epoch+1}/{num_epochs} || " + f"Loss: {loss.item()}"
                )
                print("-" * 50)
        # determine whether or not to stop early using validation set
        _, __, f1_v = validation_pytorch(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
            epoch=epoch,
            verbose=verbose,
            verbose_epoch=verbose_epoch,
        )
        if f1_v < last_metric:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}!")
                break
        else:
            trigger_times = 0
        last_metric = f1_v

    return model

def testing_pytorch(
    model: nn.Module, test_loader: DataLoader
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Evaluates the PyTorch model to a validation set and
    returns the predicted labels and their corresponding true labels

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model which inherits from the `torch.nn.Module` class
    test_loader : DataLoader
        Testing dataset as `torch.utils.data.dataloader.DataLoader` object

    Returns
    -------
    Tuple[torch.tensor, torch.tensor]
        Predicted labels and true labels
    """
    # sets the model to evaluation mode
    model.eval()
    labels_all = torch.empty((0))
    predicted_all = torch.empty((0))
    with torch.no_grad():
        # Iterate through test dataset
        for emb_t, labels_t in test_loader:
            # make prediction
            outputs_t = model(emb_t)
            _, predicted_t = torch.max(outputs_t.data, 1)
            # save predictions and labels
            labels_all = torch.cat([labels_all, labels_t])
            predicted_all = torch.cat([predicted_all, predicted_t])

    return predicted_all, labels_all

def KFold_pytorch(
    folds: KFolds,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    num_epochs: int,
    seed: Optional[int] = 42,
    patience: Optional[int] = 3,
    verbose_args: dict = {
        "verbose": True,
        "verbose_epoch": 100,
        "verbose_item": 10000,
    },
    data_loader_args: dict = {"batch_size": 1, "shuffle": True},
) -> pd.DataFrame:
    """
    Performs KFold evaluation for a PyTorch model

    Parameters
    ----------
    folds : GroupFolds
        Object which stores and obtains the folds
    model : torch.nn.Module
        PyTorch model which inherits from the `torch.nn.Module` class
    criterion : torch.nn.Module
        Loss function which inherits from the `torch.nn.Module` class
    optimizer : torch.optim.optimizer.Optimizer
        PyTorch Optimizer
    num_epochs : int
        Number of epochs
    seed : Optional[int], optional
        Seed number, by default 42
    patience : Optional[int], optional
        Patience parameter for early stopping rule, by default 3
    verbose_args : _type_, optional
        Arguments for how to print progress, by default
        {"verbose": True,
         "verbose_epoch": 100,
         "verbose_item": 10000}

    Returns
    -------
    pd.DataFrame
        Accuracy and F1 scores for each fold
    """
    torch.save(
        obj={
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "criterion": criterion,
        },
        f="starting_state.pkl",
    )
    accuracy = []
    f1_score = []
    for fold in tqdm(range(folds.n_splits)):
        print("\n" + "*" * 50)
        print(f"Fold: {fold+1} / {folds.n_splits}")
        print("*" * 50)

        # reload starting state of the model, optimizer and loss
        checkpoint = torch.load(f="starting_state.pkl")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        criterion = checkpoint["criterion"]

        # obtain test, valid and test dataloaders
        train, valid, test = folds.get_splits(fold_index=fold,
                                              as_DataLoader=True,
                                              data_loader_args=data_loader_args)

        # train pytorch model
        model = training_pytorch(
            model=model,
            train_loader=train,
            valid_loader=valid,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            seed=seed,
            patience=patience,
            **verbose_args,
        )

        # test model
        predicted, labels = testing_pytorch(model=model, test_loader=test)

        # evaluate model
        accuracy.append(((predicted == labels).sum() / labels.size(0)).item())
        f1_score.append(metrics.f1_score(labels, predicted, average="macro"))

    # remove starting state pickle file
    os.remove("starting_state.pkl")
    return pd.DataFrame({"accuracy": accuracy, "f1_score": f1_score})
