# Resnet
- Resnets utilize a **residual block**
- Normally to calculate forward propagation from $a^l$ to $a^{l+2}$, we take $a^l$, apply a linear function ($Z^{l+1}=W^{l+1}A^l+b^{l+1}$), then an activation function to get $a^{l+1}=g^{l+1}(Z^{l+1})$, then repeat to get $a^{l+2}$
    - With a residual block, we still compute the output $Z^{l+2}$ but when passing it to the activation function $g^{l+2}$, instead we pass in $Z^{l+2}+a^l$, so we get $a^{l+2}=g^{l+1}(Z^{l+2}+a^l)$
    - Also called a skip connection
- In ResNet, repeat this skip connection every 2 layers, and each pair of layers is called the **residual block**
- Allows you to train much deeper neural networks
    - Helps with exploding/vanishing gradients
    - Adding the residual block doesn't hurt performance
    - With plain NNs, neural networks have a hard time learning the identity function
- If our $A^l$ to be used in the residual block is not the same dimensions as $Z^{l+2}$, we can either zero-pad $A^l$ or generate another weight matrix $W$ that will be multiplied to $A^l$ to get the same dimensions

# 1x1 Convolutions
- If you have a $n$x$n$x$c$, you can apply a $1$x$1$x$c$ filter and convolve 
    - Can be called **network-in-network** 
        - Imagine the current slice of the block, which will be a vector; you are essentially applying a linear function to it with your filter (element-wise product of all then sum), then apply an activation function
        - Basically a layer in a regular neural network
- This can be used to change the number of channels (since we already have pooling layers)

# Inception Network
- Don't need to choose between filter sizes; we can try multiple
- Apply all filters and stack them up, channel-wise
- You can even apply max-pooling, but we need to pad the max-pooling output
## Computational Cost
- Normal convolution costs are extremely expensive
- Try using a 1x1 convolution to reduce to some lower number of channels
- Then apply regular convolution to expand back to original desired number of channels, lowers computational cost by a large amount
- Can be called a **bottleneck** layer
- Doesn't seem to hurt performance
## Inception Module
- Take an activation from layer $l$ $A^l$
- Apply a 1x1 conv layer for each size filter you want to try out
- Add a max-pool layer with padding to preserve the dimensions, then use a 1x1 conv to shrink the number of channels
- Then concatenate all results channel-wise
- A network stacks a bunch of these inception modules
    - Some inception networks have side branches that go to fully connected layers to provide softmax (seems to help avoid overfitting)