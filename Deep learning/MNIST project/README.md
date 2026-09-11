A simple MNIST project from scracht using NUMPY, for practicing purpose
## What I learned

- Forward propagation
- ReLU activation
- Softmax
- Cross-entropy loss
- Backpropagation
- Gradient descent
- One-hot encoding
- The effect of some regularization technique such as l2, Tikhonov on the accuracy rate
- *new* Using pca for dimensionality reduction to decrease the computational cost

## Model

Input (784)
   ↓
Input (50) (Dimensionality reduction by pca)
   ↓
Linear Layer(256 neurons)
   ↓
ReLU
   ↓
Linear Layer(10 neurons)
   ↓
Softmax
   ↓
Output (10)

This may have regularization, but I'll return later

## Dataset

MNIST handwritten digit dataset.

- Input size: 784 (original) -> 50
- Number of classes(output): 10

## Training

Optimizer: Gradient Descent
Loss: Cross Entropy
Activation: ReLU
