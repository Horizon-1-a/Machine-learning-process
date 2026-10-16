# A MNIST Handwritten classifier using Pytorch

This project develops MLP neural network for classifying handwritten digits from MNIST data set using Pytorch as a framework

## Project Structure

pytorch_mnist/
├── data.py            # Download and split the dataset; create DataLoaders
├── model.py           # Define the MLP architecture
├── engine.py          # Implement training and evaluation functions
├── train_model.py     # Run training and save the best checkpoint
├── requirements.txt   # List project dependencies
├── README.md          # Project documentation
├── data/              # Downloaded dataset (not tracked by Git)
└── checkpoints/       # Saved model checkpoints (not tracked by Git)

## Model Architechture

The classifier uses the following MLP architecture:

Input image (28 × 28)
        ↓
      Flatten
        ↓
  Linear (784 → 256)
        ↓
       ReLU
        ↓
  Linear (256 → 10)
        ↓
    10 logits

## Dataset

Download from MNIST dataset from torchvision


----------------------
## Result and Performance

- Model proves to gradually optimize the accuracy, starts from approximately 90% to roughly 98% after 30 epochs:
    - Best Validation accuracy: 97.46%
    - Test accuracy: 97.89%

- Overfitting has not yet systematically investigated. Need further analysis of training and validation lost and accuracy curves

## How to run

1. Clone repository
!git clone https://github.com/Horizon-1-a/Machine-learning-process.git

%cd Machine-learning-process/pytorch_learning/pytorch_mnist

2. Install dependencies

python -m pip install -r requirements.txt

3. Train the model

!python train_model.py

## Requirements:
- python 3.11.x
- pytorch
- torchvision
- numpy


