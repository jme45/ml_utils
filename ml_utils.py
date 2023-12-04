from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer
from typing import Optional, Tuple, Any

import torch
import torchmetrics
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# define additional metrics we want on top of loss. Used by TensorBoardLogger and ClassificationTrainer
additional_metrics = ("Accuracy", "F1Score")


class TensorBoardLogger:
    """
    Instance of SummaryWriter which can also be set to None.
    """

    def __init__(
        self,
        tensorboard_logging: bool,
        experiment_name: Optional[str],
        root_dir: Optional[Path] = Path("runs/"),
        additional_metrics: Optional[Tuple[str]] = additional_metrics,
    ):
        """
        Create an instance of Tensorboard SummaryWriter if tensorboard_logging is True

        :param experiment_name: str, to identify expt name
        :param root_dir: root directory for putting runs
        """
        self.additional_metrics = additional_metrics
        self.tensorboard_logging = tensorboard_logging
        self.experiment_name = experiment_name

        if tensorboard_logging:
            # Check that if we want logging, the parameters are set
            assert self.experiment_name is not None, "experiment_name must not be None"
            assert root_dir is not None, "root_dir must not be None"
            assert additional_metrics is not None, "additional_metrics must not be None"

        if tensorboard_logging:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            log_dir = root_dir / experiment_name / timestamp
            self.writer = SummaryWriter(log_dir=str(log_dir))
        else:
            self.writer = None

    def log(
        self,
        results_train: dict[str, float],
        results_test: dict[str, float],
        epoch: int,
    )->None:
        """Log results from train and test to tensorboard, if want logging"""

        # only perform logging if we actually want tensorboard logging
        if self.tensorboard_logging:
            for metric in ["loss"] + list(self.additional_metrics):
                for train_test, res in zip(
                    ["train", "test"], [results_train, results_test]
                ):
                    self.writer.add_scalar(f"{metric}/{train_test}", res[metric], epoch)

    def close(self)->None:
        '''Close the writer, if we wanted logging.'''
        if self.tensorboard_logging:
            self.writer.close()


class ClassificationTrainer:
    def __init__(
        self,
        model,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimiser_class: str | Optimizer,
        optimiser_kwargs: dict,
        loss_fn: nn.Module,
        n_epochs: int,
        device: str | torch.device,
        output_path: str | Path,
        num_classes: int,
        save_lowest_test_loss_model: bool = False,
        save_final_model: bool = False,
        tensorboard_logger: Optional[TensorBoardLogger] = None,
        disable_within_epoch_progress_bar: bool = True,
        disable_epoch_progress_bar: bool = False,
        additional_metrics: list[str] = additional_metrics,
    ) -> None:
        """
        Initialises Trainer class to train a pytorch Module.

        :param model: pytorch nn.Module to be trained
        :param train_dataloader: dataloader for training
        :param test_dataloader: dataloader for testing
        :param optimiser_class: str or instance of torch.optim, e.g. "Adam"
        :param optimiser_kwargs: kwargs for optimizer, e.g. {"lr": 1e-3}
        :param loss_fn: str or nn.Module class
        :param n_epochs: Number of epochs to run
        :param device: device on which to run, e.g. "cpu" or "cuda"
        :param save_lowest_test_loss_model: whether to save model with lowest test loss
        :param save_final_model: whether to save model obtained after max no of epochs
        :param output_path: where to save models and tensorboard logging
        :param num_classes: number of classes to classify (e.g. 5)
        :param tensorboard_logger: instance of TensorBoardLogger (could be None, no logging)
        :param disable_within_epoch_progress_bar: disable progress bar within an epoch
        :param disable_epoch_progress_bar: disable progress bar marking progress across epochs
        :param additional_metrics: metrics to log in addition to loss. Instances of torchmetrics
        :return: None
        """

        self.tensorboard_logger = tensorboard_logger
        self.disable_epoch_progress_bar = disable_epoch_progress_bar
        self.disable_within_epoch_progress_bar = disable_within_epoch_progress_bar
        self.output_path = Path(output_path)
        self.save_final_model = save_final_model
        self.save_lowest_test_loss_model = save_lowest_test_loss_model
        self.device = device
        self.n_epochs = n_epochs
        self.loss_fn = loss_fn
        self.optimiser_kwargs = optimiser_kwargs
        self.test_dataloader = test_dataloader
        self.train_dataloader = train_dataloader
        self.model = model

        # Deal with optimizer, depending on whether it's a string or the class.
        if isinstance(optimiser_class, str):
            self.optimiser = getattr(torch.optim, optimiser_class)
        self.optimiser = self.optimiser(
            params=model.parameters(), **self.optimiser_kwargs
        )

        # Set up Accuracy and F1 metric. Could implement more here. Compute on
        # CPU, as GPU won't be much faster.
        self.metrics = {
            metric: getattr(torchmetrics, metric)(
                task="multiclass", num_classes=num_classes
            ).cpu()
            for metric in additional_metrics
        }

        # If tensorboard logger not provided, initialise with no logger.
        if self.tensorboard_logger is None:
            self.tensorboard_logger = TensorBoardLogger(False, None, None, None)

        # Initialise lowest test loss and corresponding state dict. It's None initially.
        self.lowest_test_loss = None
        self.lowest_loss_state_dict = None

        # Initialise output files.
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.lowest_loss_model_path = self.output_path / "lowest_loss_model.pt"
        self.final_model_path = self.output_path / "final_model.pt"

    def _train_step(self, epoch: int) -> dict[str, float]:
        """
        Trains model through one epoch.
        :parameter: epoch: epoch which is being trained
        :return: dict of losses and performance metrics.
        """

        # Make a progress bar for progress within the epoch.
        progress_bar_generator = tqdm(
            enumerate(self.train_dataloader),
            desc=f"Training epoch {epoch}",
            total=len(self.train_dataloader),
            disable=self.disable_within_epoch_progress_bar,
        )

        # Save total number of elements across all batches.
        n_samples = 0
        total_loss = 0.0
        # save y_preds across all batches.
        y_preds_all = []
        y_true_all = []

        # set to training mode
        self.model.train()

        # Save start time, so can compute how long computation on batch took.
        start_time = timer()

        for batch, (X, y) in progress_bar_generator:
            # Save true label to cpu.
            y_true_all.append(y.cpu())

            # Send data to device.
            X, y = X.to(self.device), y.to(self.device)

            # forward pass
            y_logits = self.model(X)

            # Compute loss.
            loss = self.loss_fn(y_logits, y)

            # zero the gradients.
            self.optimiser.zero_grad()

            # backprop
            loss.backward()

            # Step the optimiser.
            self.optimiser.step()

            # Get the label predictions and append to list of all predictions.
            y_preds = torch.argmax(y_logits, dim=-1)
            y_preds_all.append(y_preds.cpu())

            # Cumulate number of samples.
            n_samples += X.shape[0]

            # Cumulate total loss.
            total_loss += loss.item()

        # Save end time.
        end_time = timer()

        # Turn y_preds_all and y_true_all into single tensor.
        y_preds_all = torch.cat(y_preds_all)
        y_true_all = torch.cat(y_true_all)

        # Make a dictionary with metrics for returning.
        ret_metrics = {}
        # Save average loss per sample.
        ret_metrics["loss"] = total_loss / n_samples
        for metric, fn in self.metrics.items():
            ret_metrics[metric] = fn(y_preds_all, y_true_all)

        # Save training time.
        ret_metrics["epoch_time"] = end_time - start_time

        return ret_metrics

    def _test_step(self, epoch):
        # Make a progress bar for progress within the epoch.
        progress_bar_generator = tqdm(
            enumerate(self.test_dataloader),
            desc=f"Testing epoch {epoch}",
            total=len(self.test_dataloader),
            disable=self.disable_within_epoch_progress_bar,
        )

        # Save total number of elements across all batches.
        n_samples = 0
        total_loss = 0.0
        # Save y_preds across all batches.
        y_preds_all = []
        y_true_all = []

        start_time = timer()

        # Put model in evaluation mode.
        self.model.eval()

        for batch, (X, y) in progress_bar_generator:
            # Save true label
            y_true_all.append(y.cpu())

            # Send data to device.
            X, y = X.to(self.device), y.to(self.device)

            with torch.inference_mode():
                # forward pass
                y_logits = self.model(X)

                # compute loss
                loss = self.loss_fn(y_logits, y)

            # Get the label predictions
            y_preds = torch.argmax(y_logits, dim=-1)

            y_preds_all.append(y_preds.cpu())

            # cumulate number of samples
            n_samples += X.shape[0]

            # cumulate total loss
            total_loss += loss.item()

        end_time = timer()

        # Turn y_preds_all and y_true_all into single tensor.
        y_preds_all = torch.cat(y_preds_all)
        y_true_all = torch.cat(y_true_all)

        # Make a dictionary with metrics for returning.
        ret_metrics = {}
        ret_metrics["loss"] = total_loss / n_samples
        for metric, fn in self.metrics.items():
            ret_metrics[metric] = fn(y_preds_all, y_true_all)

        # Save training time.
        ret_metrics["epoch_time"] = end_time - start_time

        return ret_metrics

    def train(self) -> dict[str, list[float]]:
        """
        Train the model.
        :return: dict with training metrics.
        """
        epoch_generator = tqdm(
            range(self.n_epochs),
            desc=f"Training loop",
            total=self.n_epochs,
            disable=self.disable_epoch_progress_bar,
        )

        # make a dictionary containing all results
        all_results = {"train_loss": [], "test_loss": []}
        for train_test in ["train", "test"]:
            for metric in self.metrics.keys():
                all_results[f"{train_test}_{metric}"] = []
            all_results[f"{train_test}_epoch_time"] = []

        # Iterate through epochs
        for epoch in epoch_generator:
            results_train = self._train_step(epoch)
            results_test = self._test_step(epoch)

            # add all results to dictionary.
            for train_test, res in zip(
                ["train", "test"], [results_train, results_test]
            ):
                for metric in res.keys():
                    all_results[f"{train_test}_{metric}"].append(res[metric])

            # Check for lowest test loss. If lower than previous one, save state dict.
            if self.lowest_test_loss is None or (
                results_test["loss"] < self.lowest_test_loss
            ):
                self.lowest_loss_state_dict = self.model.state_dict()

            self.tensorboard_logger.log(results_train, results_test, epoch)

        # Save state dict of models we want to save.
        if self.save_lowest_test_loss_model:
            torch.save(obj=self.lowest_loss_state_dict, f=self.lowest_loss_model_path)
        if self.save_final_model:
            torch.save(obj=self.model.state_dict(), f=self.final_model_path)

        # need to close the tensorboard writer.
        self.tensorboard_logger.close()

        return all_results
