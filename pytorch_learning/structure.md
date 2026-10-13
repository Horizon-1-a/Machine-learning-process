# grad:
- pytorch automatically saves the value of gradient of each variables when apply y.backward()
- Applying gradient descend requires setting gradient to be zero if I train model manually:
  "with torch.no_grad()"
# nn library:
  ## optimizer:
  - An object retain group of model to train and the optimizing algorithms using for gradient descend
  - Do not need to call backward because of autograd, but still have to call zero_grad()
  - Updating by the call .step()
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
  ## model.train(), model.eval(), no_grad():
  
  ### training and evaluation mode:
    - Train mode is used for training model, with some layers are tailored for training (Dropouts, Batchnorm, ...)
    - Eval mode switch those layers into mode serving for inference purpose(Only apply to layers in specific model, not switch off autograd or computation graph)
  ### no_grad:
    - Temporarily turn of autograd engine for evaluation purpose, ...

    ---> To evaluate model, it is essential to combine both eval() and no_grad() 

  ## Dataset, Dataloader:

  ### Dataset:
  - A class allow access to input data and corresponding output data.
      ### Reason:
        - A class enable access to data by file path, without having saved data direcly into Memory
        - Can modify data before processing, achieve multiple result in the same epoch
        - Synthesize with dataloader to take advantage of num-works
  ### Dataloader:
  - Turn Dataset into iterable version of mini-batch, also take resposible for shuffle and other data loading thing

  ## Loss function:
  - Having the same operation as from-scratch version
  - Paying attention to container format: np array >< tensor


    
    
  


  
