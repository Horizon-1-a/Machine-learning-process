**convolution network has format conv2d(out_feat, in_feat, stride, padding)**

## Shape
 ### Weight tensor :(C_out, C_in, H, W)
 - C_in : num of channel in (num of kernel in each filter)
 - C_out :num of channel out(num of filter out)
 - (H, W): shape of kernel

    Operation : 
    - Each filter has c_in kernel, which is corresponding to the number of in_channels
    - Each kernel applies conv to its corresponding channel, results in new feature map
    - Adding element-wise feature maps above, we have new feature map of one filter
 ### Image tensor: (N, C, H, W):
 - N : batch size
 - C : num of channels
 - (H, W) : Shape

## stride, padding
 ## stride 
 - Step width which kernel travels in each iteration
 ## padding
 - Adding an 0 layer along the edge of image (or input)

formulae: out_shape = (n + 2 * p - k) // s + 1

## Code
see code/numpy_imple.py