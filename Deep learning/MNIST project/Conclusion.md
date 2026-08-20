With the model has these specific configurations:
Input: 784
Hidden: 256 ReLU
Output: 10 Softmax

Optimizer:
Mini-batch Gradient Descent
Batch size: 50

Training:
Epochs: 50
Learning rate: ...
Training samples: ...

--> Results:
Best validation accuracy: 94.4%
Test accuracy: 95.12%

## Observation:
While the training accuracy gradually increase throughout the training period, the validation accuracy saturates at about 95%, idicating
that the model is trying to remember the training dataset instead of predict the answer.
