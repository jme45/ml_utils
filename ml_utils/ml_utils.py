from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer
from typing import Optional, Tuple, Any, Callable
from contextlib import nullcontext

import torch
import torchmetrics
from torch import nn
from torch import Tensor
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
        root_dir: Optional[Path] = Path("runs/"),
        additional_metrics: Optional[Tuple[str]] = additional_metrics,
    ):
        """
        Create an instance of Tensorboard SummaryWriter if tensorboard_logging is True

        :param root_dir: root directory for putting runs
        """
        self.additional_metrics = additional_metrics
        self.tensorboard_logging = tensorboard_logging

        if tensorboard_logging:
            # Check that if we want logging, the parameters are set
            assert root_dir is not None, "root_dir must not be None"
            assert additional_metrics is not None, "additional_metrics must not be None"

            log_dir = Path(root_dir) / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
        else:
            self.writer = None

    def log(
        self,
        results_train: dict[str, float],
        results_test: dict[str, float],
        epoch: int,
    ) -> None:
        """Log results from train and test to tensorboard, if want logging"""

        # only perform logging if we actually want tensorboard logging
        if self.tensorboard_logging:
            for metric in ["loss"] + list(self.additional_metrics):
                for train_test, res in zip(
                    ["train", "test"], [results_train, results_test]
                ):
                    self.writer.add_scalar(f"{metric}/{train_test}", res[metric], epoch)

    def close(self) -> None:
        """Close the writer, if we wanted logging."""
        if self.tensorboard_logging:
            self.writer.close()


def default_state_extractor_function(model: nn.Module) -> dict[str, Any]:
    """
    Function for obtaining state dict of a module.

    Sometimes we may only want to get part of a module. This default function gets
    the entire state dict.
    :param model:
    :return: state_dict of the module in the form of an OrderedDict
    """
    return model.state_dict()


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
        print_progress_to_screen: bool = False,
        state_dict_extractor: Callable[
            [nn.Module], dict[str, Any]
        ] = default_state_extractor_function,
        trainable_parts: str | list[str] = "all",
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
        :param print_progress_to_screen: whether to print loss and accuracy to screen.
        :param state_dict_extractor: Function to extract appropriate state dict from model.
        :param trainable_parts: which parts of model are trainable. Default: "all"
        :return: None
        """
        self.trainable_parts = trainable_parts
        self.state_dict_extractor = state_dict_extractor
        self.print_progress_to_screen = print_progress_to_screen
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
            self.tensorboard_logger = TensorBoardLogger(
                False,
                None,
                None,
            )

        # Initialise lowest test loss and corresponding state dict. It's None initially.
        self.lowest_test_loss = None
        self.lowest_loss_state_dict = None

        # Initialise output files.
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.lowest_loss_model_path = self.output_path / "lowest_loss_model.pth"
        self.final_model_path = self.output_path / "final_model.pth"

        # Check that compute accuracy if printing to screen, since this assumes Accuracy is present.
        if self.print_progress_to_screen:
            assert (
                "Accuracy" in additional_metrics
            ), "If printing to screen, 'Accuracy' must be part of additional_metrics."

    def _assemble_ret_metrics(
        self,
        y_preds_all: list[Tensor],
        y_true_all: list[Tensor],
        total_loss: float,
        n_samples: int,
        start_time: float,
        end_time: float,
    ) -> dict[str, float]:
        # Turn y_preds_all and y_true_all into single tensor.
        y_preds_all = torch.cat(y_preds_all)
        y_true_all = torch.cat(y_true_all)

        # Make a dictionary with metrics for returning.
        ret_metrics = {}
        # Save average loss per sample.
        ret_metrics["loss"] = total_loss / n_samples
        for metric, fn in self.metrics.items():
            ret_metrics[metric] = fn(y_preds_all, y_true_all).item()

        # Save training time.
        ret_metrics["epoch_time"] = end_time - start_time
        return ret_metrics

    def _set_model_to_test_or_eval(self, train_test):
        """
        Set the parts which are trainable to test or eval

        Sometimes we don't want to train the entire model, in that
        case not the entire model is set to train/eval, but only
        the trainable sub-parts.
        """
        if train_test == "train":
            if self.trainable_parts == "all":
                self.model.train()
            else:
                for trainable_part in self.trainable_parts:
                    getattr(self.model, trainable_part).train()
        elif train_test == "test":
            if self.trainable_parts == "all":
                self.model.eval()
            else:
                for trainable_part in self.trainable_parts:
                    getattr(self.model, trainable_part).eval()
        else:
            raise NotImplementedError(
                f"train_test must be 'train' or 'test'. Got train_test={train_test}"
            )

    def _train_or_test_step(self, train_test: str, epoch: int):
        """
        One function, which either does test or train step.

        Test and train steps are so similar, that it's better to have a single
        function which can do both.
        :param test_train: "test" or "train", depending on which step is being done
        :param epoch: epoch which is being trained
        :return: dict of losses and performance metrics.
        """
        assert train_test in [
            "train",
            "test",
        ], "test_train must either be 'test' or 'train'"

        # Get correct dataloader.
        if train_test == "train":
            dataloader = self.train_dataloader
        else:
            dataloader = self.test_dataloader
        self._set_model_to_test_or_eval(train_test)

        # Make a progress bar for progress within the epoch.
        progress_bar_generator = tqdm(
            enumerate(dataloader),
            desc=f"{train_test.title()}ing epoch {epoch}",
            total=len(dataloader),
            disable=self.disable_within_epoch_progress_bar,
        )

        # Save total number of elements across all batches.
        n_samples = 0
        total_loss = 0.0
        # save y_preds across all batches.
        y_preds_all = []
        y_true_all = []

        # Save start time, so can compute how long computation on batch took.
        start_time = timer()

        for batch, (X, y) in progress_bar_generator:
            # Save true label to cpu.
            y_true_all.append(y.cpu())

            # Send data to device.
            X, y = X.to(self.device), y.to(self.device)

            # Inference mode only needed during testing.
            with torch.inference_mode() if train_test == "test" else nullcontext():
                # forward pass
                y_logits = self.model(X)

                # Compute loss.
                loss = self.loss_fn(y_logits, y)

            # Optimisation steps only needed during training.
            if train_test == "train":
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

        ret_metrics = self._assemble_ret_metrics(
            y_preds_all, y_true_all, total_loss, n_samples, start_time, end_time
        )

        return ret_metrics

    def _print_progress_to_screen(
        self,
        results_train: dict[str, float],
        results_test: [str, float],
        epoch: int,
    ):
        """Print loss and accuracy to screen"""
        if self.print_progress_to_screen:
            items_to_print = []
            for train_test, res in zip(
                ["train", "test"], [results_train, results_test]
            ):
                for metric in ["loss", "Accuracy"]:
                    items_to_print.append(f"{train_test} {metric}: {res[metric]:0.3f}")
            print(", ".join(items_to_print))

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

        # Send the model to the device.
        self.model = self.model.to(self.device)

        # Iterate through epochs
        for epoch in epoch_generator:
            results_train = self._train_or_test_step("train", epoch)
            results_test = self._train_or_test_step("test", epoch)

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
                self.lowest_loss_state_dict = self.state_dict_extractor(self.model)

                # Save model as soon as it's computed, in case programme doesn't terminate.
                if self.save_lowest_test_loss_model:
                    torch.save(
                        obj=self.lowest_loss_state_dict, f=self.lowest_loss_model_path
                    )

            self._print_progress_to_screen(results_train, results_test, epoch)
            self.tensorboard_logger.log(results_train, results_test, epoch)

        # Save state dict of models we want to save.
        if self.save_lowest_test_loss_model:
            torch.save(obj=self.lowest_loss_state_dict, f=self.lowest_loss_model_path)
        if self.save_final_model:
            torch.save(
                obj=self.state_dict_extractor(self.model), f=self.final_model_path
            )

        # need to close the tensorboard writer.
        self.tensorboard_logger.close()

        return all_results
