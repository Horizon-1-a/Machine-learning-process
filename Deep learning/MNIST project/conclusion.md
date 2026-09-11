**With the model using the following configurations:**

Input: 784

Hidden: 256 ReLU

Output: 10 Softmax

I have observed some effects:
## 1.Regularization
### 1.1.L2 regularization

- The use of l2 regularization indeed enhance the accuracy rate of the model after training by penalizing the large weights and therefore reducing overfitting
- This method depends on the hperparameter lamda
- If I increase the lamda too much, the model may underfitting, which is worsening the accuracy rate
- On the other hand, the small lamda may negligimate the effect

### 1.2 Tikhonov regularization
- This is the generalization of l2
- In my experiment, I randomly generated T\, this can not improve the performance compared to the l2 version
- Indeed, the random T cannot capture the full meaning or prior of the problem
- Learning how to design this may require many time, so I leave to the future
## 2.Gradient descent
- Implying some methods like SGD or mini-batch gradient descent can accelerate the minizing loss function process
- By dividing them into smaller part, the parameter can update more frequent and reducing the computational time
## 3. Training set, validation set, test set
- Splitting the original training set onto training and validation set. The former is training material, the later is a benchmark
- The common ratio is 8:2, but it still depend on the original dataset
  
  ### 3.1 Early stopping
  - Using to save the model having the best perfomace having the highest validation accuracy rate, preventing them from reaching overfitting
  - Because the validation set may have the same meaning as the test set, this method can enhance the output considerably
## *new* 4.PCA
- Reducing the capacity of model, therefore lower the computational cost
- Need to consider trade-off: Accuracy >< Capacity

  
  



  




