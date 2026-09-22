# grad:
- pytorch automatically saves the value of gradient of each variables when apply y.backward()
- Applying gradient descend requires setting gradient to be zero if I train model manually:
  "with torch.no_grad()"
# optimizer:
- An object retain group of model to train and the optimizing algorithms using for gradient descend
- Do not need to call backward because of autograd, but still have to call zero_grad()
- Updating by the call .step()
# module:
- A abstaction for one component of a neural network, including the submodels and the activities of them.
** An analogy:
  If we consider neural network as multiple of many mappings, the modules is the mapping afterward, F
  
