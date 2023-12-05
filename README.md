# ml_utils
> Code for training pytorch modules


## Installation
```
pip install <directory of source code>
```

## Usage example

```python
import ml_utils
from torch import nn


trainer = ml_utils.ClassificationTrainer(model, 
                                         train_dataloader, 
                                         test_dataloader, 
                                         "Adam", 
                                         {"lr": 1e-3}, 
                                         nn.CrossEntropyLoss(),
                                         n_epochs=2,
                                         output_path='run1',
                                         device='cpu',
                                         num_classes=2,
                                         )

results = trainer.train()
```