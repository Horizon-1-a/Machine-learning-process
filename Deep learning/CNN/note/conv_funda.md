## Motivation : Why CNN?

- An MLP can learn how to classify image, but some important aspects in detecting data, such as edge, etc cannot be saw by MLP due to flattened data and fully connected layer. This may hinder the performance of MLP as it can learns how to do task by only revising a huge number of training sample
- Also, when the size of image become larger, the number of parameter in MLP can become extremely large, which can cause poor performance and overfiting if training dataset is small

CNN introduces these suitable feature:
- Local connectivity: kernel process a local region
- weight sharing: same kernel is applied to many region

This allows CNN to detect patterns at different position of image

## Mental model
 kernels slide across the input image

- Take a local batch
- each kernel multipies with its corresponding channel (ex: channel 3 if RGB)
- Sum over feature maps
- Store result as a new feature map 

## Numpy implementation

See (code/numpy_imple.py)
