# Vấn đề
Khi xây dựng một mô hình, có thể sẽ gặp hiện tượng training error rất thấp, nhưng test error lại rất cao. Điều đó có thể do hiện tượng overfitting gây ra, đó là việc mô hình có cấu trúc phức tạp nên sẽ rất sát với training set dẫn đến việc dự đoán các output của test data có nhiều sai lệch.

## Các tập dữ liệu cần biết
**Training set**: Set using for training the model

**Validation set**: Set extracted from the original training set, independent from the current training set. Using to determine whether the model has the good estimation without using the test set(Has the same meaning with the test set

**Test set**: Using to estimate the ability of model
   - Purpose: Minimizing the training error and validation error

## Solution

### 1.Validation:

- Spitting the training dataset into two disjoint set: training and validation, then trying to minimize the validation error
  
### 2.Regularization

- Adjusting model to avoid overfitting, that is reducing the complexity of the model, with the slight increase of loss as a trade-off
  
**2.1** L2 regularization

- Algorithm: Add the norm 2 of weight when computing the dz
  
  * [Ex](https://machinelearningcoban.com/2017/03/04/overfitting/#-%5C%5Cl%5C%5C-regularization)
  
- Meaning:l reducing the size of the weight by fining the ones having large absolute value
**2.2** Tikhonov regularization
  
  A general version of l2 regularization
  
Instead of using the bias as in the l2 regularization, this method using a (diagonal) matrix when computing the loss function

**2.3** L0 regularization 

The norm 0 version of l2 regularization

I'll get dig into it later

There are many type of regularization used regularly such as dropout,..., and I'll learn later






