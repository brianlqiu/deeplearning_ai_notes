# Computer Vision
- Computer vision problems
    - Image classification
    - Object detection/location
    - Neural style transfer
- Neural networks on large images are difficult because too many parameters

## Edge Detection
- Detect vertical and horizontal edges
- Look at grayscale
- Input the grayscaled image and convolve it with a filter (matrix), denoted by $*$
- Example: Take a 6x6 grayscaled image, try to compute vertical edge
    - We use a filter of 3x3 $\begin{bmatrix}1&0&-1\\1&0&-1\\1&0&-1\end{bmatrix}$
    - Convolute the 6x6 with the 3x3 matrix
    - Start with the filter over the top left corner of the 6x6 matrix
    - Compute the element-wise product of the part of the matrix covered by the filter and the filter, then sum the element-wise product into one number
    - This will be the top left element in our resulting 4x4 matrix
        - 4 rows and 4 columns since you can move your filter at most 4 times to the right to cover all possibilities
    - Creates a new "image"
    - Imagine example where we have grayscale image where >0 = lighter and <0 = darker, and we have matrix $\begin{bmatrix}1&1&1&0&0&0\\1&1&1&0&0&0\\1&1&1&0&0&0\\1&1&1&0&0&0\\1&1&1&0&0&0\\1&1&1&0&0&0\\\end{bmatrix}$
    - If we apply our example 3x3 filter to the image, we get a result of $\begin{bmatrix}0&3&3&0\\0&3&3&0\\0&3&3&0\\0&3&3&0\\\end{bmatrix}$ 
    - The example 3x3 filter does bright-to-dark filtering, if we care about dark-to-bright we can rotate 180 or just take the absolute value
- If we want horizontal edge detection, we can rotate the matrix 90 degrees: $\begin{bmatrix}1&1&1\\0&0&0\\-1&-1&-1\end{bmatrix}$
- Convolution operator depends on language/framework

### Filters
- Different filters have been debated on
- Sorbel filter - more weight towards the middle rows
$\begin{bmatrix}1&0&-1\\2&0&-2\\1&0&-1\end{bmatrix}$
- Charr filter 
$\begin{bmatrix}3&0&-3\\10&0&-10\\3&0&-3\end{bmatrix}$
- Or we can train the filter as parameters, which can be more specific to the data

```
conv-forward    # Python
tf.nn.conv2d    # Tensorflow
Conv2D          # Keras
```

## Padding 
- If we convolute a $n$x$n$ filter with a $f$x$f$ filter, the resulting matrix will be $(n-f+1)$x$(n-f+1)$
    - Can't convolute too many times, otherwise image will get too small
- Also, edge pixels won't be accounted for as much as the middling images
- Solve by **padding** the border with 0s, can solve shrinking issue and also use more of the edge data
- With padding of $p$, our resulting will be $(n+2p-f+1)$x$(n+2p-f+1)$
- **Valid** convolutions - no padding
- **Same** convolutions - pad so output size is same as input size
    - $p=\frac{f-1}2$
    - $f$ is almost always odd

## Strided Convolutions
- **Stride** - the number of steps you take when moving the filter
- Resulting matrix will be $(\lfloor\frac{n+2p-f}s+1\rfloor)$x$(\lfloor\frac{n+2p-f}s+1\rfloor)$
    - We need floor if stride goes outside of the image

## Convolutions on RGB Images
- If we have a $n$x$n$x$3$ image, we'll need a $f$x$f$x$3$ filter, which will produce a $4$x$4$x$1$ image
- Same convolution operator, now we're just doing it with 3D matrices instead of 2D
    - Can customize filter to recognize certain colors by altering the filter of each channel

## Multiple Filters
- If we convolve with multiple filters, we can stack the different outputs to create a 3D matrix 
    - If we get 2 4x4 matrices through 2 different filters, we can get a 4x4x2 resulting matrix
- Number of channels can be called depth

## Layer
- First, apply $n_f$ filters to get $n_f$ outputs
- Then add a constant weight $b_i$ to correspond with the $i$th filter
    - $b_i$ will be different for each filter
- Then apply a linearity/activation function to each weighted output
- Then stack the weighted & activated outputs, producing one layer
### Notation
- If layer $l$ is a convolution layer
    - $f^{[l]}$ is the filter size
    - $p^{[l]}$ is the padding
    - $s^{[l]}$ is the stride
    - $n^{[l]}_H$ is the height of the output of this layer
    - $n^{[l]}_W$ is the width of the output of this layer
    - $n^{[l]}_C$ is the channel of the output or the number of filters of this layer
- $n^{[l]}_H=n^{[l]}_W=\lfloor\frac{n^{[l-1]}_{H/W}+2p^{[l]}-f^{[l]}}{s^{[l]}}+1\rfloor$
- Each filter is $f^{[l]}$x$f^{[l]}$x$n_c^{[l-1]}$

## Convolutional Network Layers
- Usually we feed image through CNN, then to logistic regression
- Types of layers:
    - Convolution
    - Pool
    - Fully connected

### Pooling Layers
- **Pooling layers** reduce the size of representation to speed up the computation and makes the features more robust
- **Max pooling** - use a filter size $f$ and stride $s$ to take the max value of each filter region
    - When applying max-pooling for mutliple channels/filters, compute max-pooling independently on each channel to preserve channel number/depth
- **Average pooling** - like max, but with averages
    - Max pooling usually works better, but sometimes used for last layer
- Changing the $f$ and $s$ changes the factor at which the input shrinks
- Padding is rarely or never used
- No parameters to learn

## CNN Example
- We want to recognize handwritten numbers as 32x32x3 image
- Each layer is composed of a convolutional layer and a pool layer
- After final conv-pool layer, we can flatten out it into a vector for input into a regular NN
- Layers of the regular NN are called fully-connected
- Since we need to classify 0-9, output will be softmax
- Notes:
    - As the image passes through the conv-pool layers, the height & width decrease while the depth/channels increase

## Why Convolutions?
- Parameter sharing - feature detector that's useful in one part of the image is probably useful in another part of the image
- Sparsity of connections - in each layer, each output depends only on a small number of inputs
- Less prone to overfitting
- Better with shifted images
