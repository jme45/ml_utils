# ChangeLog

## [Unversioned]

## [0.2.0] - 2024-01-12
- Converted to a proper package.
- Combined _test_step and _train_step into one function, _test_or_train_step.
- Added test for checking that random data for test remains constant.
- Removed experiment_name from TensorboardLogger.
- Changed model output file name extension from .pt to .pth.
- Changed rootdir behaviour for tensorboard logging.
- Implemented possibility of passing in function for extracting state_dict
- Implemented possibility to only set some parts of the model to .train() or .eval()

## [0.1.0] - 2023-12-05
- Code works for basic use. Tests implemented.