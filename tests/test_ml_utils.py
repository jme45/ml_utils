import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from ml_utils_jme45 import ml_utils


def test_trainer(tmp_path):
    """
    Build  a simple 3 layer network and train it over 1 epoch.
    Check output matches expected output.
    Use random seed to produce reproducible code.
    :return:
    """
    n_samples = 4
    n_test_samples = 2
    n_input_features = 3
    n_classes = 3
    n_hidden_features = 4
    # Set random seed for reproducibility.
    torch.manual_seed(0)

    model = nn.Sequential(
        nn.Linear(n_input_features, n_hidden_features),
        nn.ReLU(),
        nn.Linear(n_hidden_features, n_classes),
    )

    X = torch.randn(n_samples, n_input_features)
    y = torch.randint(0, n_classes, (n_samples,))
    X_test = torch.randn(n_test_samples, n_input_features)
    y_test = torch.randint(0, n_classes, (n_test_samples,))

    # Check that at least some of the inputs match, otherwise
    # there is no chance that the outputs will match.
    expected_X = np.array(
        [
            [-0.5655086, 0.16043702, -0.02535119],
            [1.0739002, 2.2628458, -0.9175293],
            [-0.22511572, 2.3466382, -1.1088485],
            [1.6253527, 1.2333362, -0.18318552],
        ]
    )
    assert np.allclose(expected_X, X.numpy()), "Random input data doesn't match."

    train_dl = DataLoader(TensorDataset(X, y), 2, False)
    test_dl = DataLoader(TensorDataset(X_test, y_test), 1, False)

    output_path = tmp_path / "runs"

    trainer = ml_utils.ClassificationTrainer(
        model,
        train_dl,
        test_dl,
        "SGD",
        {"lr": 0.1},
        nn.CrossEntropyLoss(),
        n_epochs=2,
        device="cpu",
        output_path=output_path,
        num_classes=n_classes,
        save_lowest_test_loss_model=False,
        save_final_model=True,
        disable_within_epoch_progress_bar=True,
        disable_epoch_progress_bar=True,
    )

    res = trainer.train()

    # Remove timing information, as that will not remain constant.
    res = {k: v for k, v in res.items() if "time" not in k}

    # Convert to pandas.
    res_df = pd.DataFrame(res)

    # Get expected values as numpy.
    expected = np.array(
        [
            [0.49871677, 1.32429612, 0.5, 0.5, 0.0, 0.0],
            [0.46373421, 1.37343222, 0.75, 0.75, 0.0, 0.0],
        ]
    )

    assert np.allclose(res_df.values, expected), "Output doesn't match."
