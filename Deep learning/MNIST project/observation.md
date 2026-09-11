## 1.Experimental setup

### Dataset

- Dataset: MNIST
- Training samples: 48,000
- Validation samples: 12,000
- Test samples: 10,000
- 
### Preprocessing

- Pixel values are normalized to [0, 1].
- Training mean is calculated only from the training set.
- The same mean is applied to validation and test sets.
- PCA is fitted only on the training set.

### Model

- Hidden layer: 256 neurons
- Hidden activation: ReLU
- Output activation: Softmax
- Batch size: 50
- Epochs: 50
- Learning rate: 0.1
- Random seed: 42
## 2.Experiment

### Experiment 1: MLP without pca

#### Results

| Metric | Value |
|---|---:|
| Final train accuracy | 100% |
| Best validation accuracy |  |
| Test accuracy | 97% |

#### Observation:

The gap between the train accuracy and test accuracy shows the insufficient in generalizing problem of model, that is overfitting

### Experiment 2: MLP with pca

#### Configuration: 

Same as Ex.1 + reduce x dimension to k

#### Result:

| Number of components | Train accuracy | Validation accuracy |
|---:|---:|---:|
| 784 | 100 | 97 |
| 200 | ... | ... |
| 100 | ... | ... |
| 50 | ... | ... |
| 20 | ... | ... |

