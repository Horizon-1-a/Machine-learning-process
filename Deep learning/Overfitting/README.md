# Vấn đề
Khi xây dựng một mô hình, có thể sẽ gặp hiện tượng training error rất thấp, nhưng test error lại rất cao. Điều đó có thể do hiện tượng overfitting gây ra, đó là việc mô hình có cấu trúc phức tạp nên sẽ rất sát với training set dẫn đến việc dự đoán các output của test data có nhiều sai lệch.

# Các tập dữ liệu cần biết
* Training set: Set using for training the model
* Validation set: Set extracted from the original training set, independent from the current training set. Using to determine whether the model has the good estimation without using the test set(Has the same meaning with the test set
* Test set: Using to estimate the ability of model
   --> Purpose: Minimizing the training error and validation error

## Solution
* Validation:
- Spitting the training dataset into two disjoint set: training and validation, then trying to minimize the validation error
* Regularization
- Adjusting model to avoid overfitting, that is reducing the complexity of the model, with the slight increase of loss as a trade-off
  Some regularization techniques:



