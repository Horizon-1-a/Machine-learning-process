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

## Model

Input (784)
   ↓
Linear Layer
   ↓
ReLU
   ↓
Linear Layer
   ↓
Softmax
   ↓
Output (10)

## Dataset

MNIST handwritten digit dataset.

- Input size: 784
- Number of classes: 10

## Training

Optimizer: Gradient Descent
Loss: Cross Entropy
Activation: ReLU
