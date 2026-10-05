# grad:
- pytorch automatically saves the value of gradient of each variables when apply y.backward()
- Applying gradient descend requires setting gradient to be zero if I train model manually:
  "with torch.no_grad()"
# optimizer:
- An object retain group of model to train and the optimizing algorithms using for gradient descend
- Do not need to call backward because of autograd, but still have to call zero_grad()
- Updating by the call .step()
# nn library:
  ## module:
  - A abstaction for one component of a neural network, including the submodels and its feature
  
  Model tree:
  
  Model |-> submodel |->....
        |            |->....
        |-> Features(parameter, ...)
  ## Linear:
  - Doing the same operation as an linear function
  - Receive (out_feature, in_feature) as input
  ** Caution: This module's forward uses for sample x feature dataset, so need to take the transpose the data before use and transpose the result to get the desired results.
  ## Sequential:
  - Register sequence of module + determine the computation order when calling forward
  ## State_dict():
  Clarification: Architecture, Parameters, State:
    - Architecture: Describe the structure and the computation of model, such as submodel and how they are linked.
    - Parameter: Tensors that are usually registered to learn through training
    - Model State: Preserve parameter, with persistent buffer which not update throughout training period. The key-value show their relation in the module tree
    + Optimizer state: Preseve state and configuration optimizer need for optimizing operation, with hyperparameter 
    
  


  
