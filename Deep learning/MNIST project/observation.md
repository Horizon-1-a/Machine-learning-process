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

Same as Ex.1 + reduce data dimension to k

#### Result:

| Number of components | Train accuracy | Validation accuracy |
|---:|---:|---:|
| 784 | 100 | 97.8 |
| 200 | ~100 | 97.62 |
| 100 | ~100 | 97.21 |
| 50 | ~100 | 97.6 |
| 20 | 99.08 | 97.17 |

#### Observation

Reducing number of dimension decreases the accuracy.
However, this can lower the capacity of model
--> Trade-off: accuracy <> capacity

### Experiment 3: MLP with pca(50) and regularization

#### Configuration:

Same + pca(50) + regularizaton

#### Result

| Lambda | Train accuracy | Validation accuracy | Weight norm |
|---:|---:|---:|---:|
| 1e-6 | ~100 | 97.43 | 2  |
| 1e-7 |~100| 97.37 | 2 |
| 0.001 | 98.48 | 97.29 | 2 |
| 1.0 | 11.23 | 11.35 | 2|

#### Observation:
- Small lambda may encourage the accuracy rate to increase
- Too large lambda may result in underfitting


