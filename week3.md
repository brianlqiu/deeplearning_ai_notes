# Neural Networks Overview
- Like in the logistic regression where it could be represented as a computation graph, a neural network is composed of layers of these nodes connected
- In previous we only have one node with $z=w^Tx+b \rightarrow a=\sigma(z)$, now we can have layers of these nodes feeding their output to other nodes
- Notation: use square brackets to denote quantities related to a layer ($W^{[i]}$) 

# Neural Network Representation
- Input layer
    - Whereas previously our examples were called X, we now pass on $a^{[0]}$ which are the activations for layer 0
- Multiple hidden layers
    - Assuming $n-1$ hidden layers, the activations for each layer is denoted $a^{[1]}$ to $a^{[n]}$
        - These layers are vectors
    - The activation for a specific node is denoted $a^{layer}_{index}$
- Output layer
- Input layer not counted in the name, just number of hidden + output

# Computing a Neural Network's Output
- Imagine you have a 3 inputs, 4 nodes in a hidden layer, and a single node output layer
- Each node computes its own $z^{[1]}_n=w^{[1]T}_nx+b^{[1]}_1, a^{[1]}_n=\sigma(z^{[1]}_n)$
- We can vectorize this computation by stacking these 4 $w^{[1]}$ into a single matrix of size $(4,3)$ (4 nodes, 3 inputs)
- Then $w^{[1]T}x$ will produce a $(4,1)$ matrix, which means we will also need $b^{[1]}$ to be a $(4,1)$ matrix for addition 
    - Note that each node has it's own $w$ and $b$
- Then computing $a=\sigma(z)$ produces a $(4,1)$ matrix $a$ with the activations of each node
- Steps to computing one example summarized as follows:
    - $z^{[1]}=W^{[1]}x+b^{[1]}$
    - $a^{[1]}=\sigma(z^{[1]})$
    - $z^{[2]}=W^{[2]}a^{[1]}+b^{[2]}$
    - $a^{[2]}=\sigma(z^{[2]})$

# Vectorizing Across Multiple Examples
- Previous section only computes for one example
- Use the same method, except now $X$ is a $(n,m)$ matrix (n features, m examples), so $Z^{[1]}$ will become a $(l,m)$ matrix, where $l$ is the number of nodes in the layer and $m$ is the number of examples
- Since sigmoid function is applied elemnt wise, $A^{[1]}$ is of the same dimension
- Then $Z^{[2]}$ and $A^{[2]}$ will be $(1,m)$ matrices, since $W^{[2]}$ is $(1,l)$ and $A^{[1]}$ is $(l,m)$, which holds the $m$ probabilities for each example

# Activation Functions #
- We've been using sigmoid functions, but other functions can be better
- $\tanh(z)$ usually works much better than sigmoid
    - Just shifted sigmoid, since it works better with the standardized dataset
    - Sigmoid might be used for output layer since it produces a value between 0 and 1
- Problem with sigmoid and tanh functions is that if the calculated $z$ is very large or very small, the gradient becomes very small and it takes a long time to converge
- Rectified linear unit (ReLU) is very popular
    - $a=max(0,z)$
- For binary classification in output layer, sigmoid can be a good choice
- Usually all other node you should use RELU
- Leaky ReLU is like ReLU but instead of being 0 when $z$ is negative, we get a smaller upwards slope instead
    - Usually better than ReLU, but not implemented as often
## Pros and Cons of Activation Functions
- Sigmoid: $a=\frac1{1+e^{-z}}$
    - Consider using only for binary classification output layer
- Tanh: $a=\frac{e^z-e^{-z}}{e^z+e^{-z}}$
    - Superior version of sigmoid, but overshadowed by other functions
- ReLU: $a=\max(0,z)$
    - The default activation function to use
- Leaky ReLU: $a=\max(0.001z,z)$
    - A better but less popular version of ReLU
    - Slope can be any small value
## Why non-linear activation functions?
- All you get is a linear function at the end
- However, you might use a linear activation function at the output if the data fits some linear trend

# Derivatives of Activation Functions
- Sigmoid: $\sigma'(x)=\sigma(x)(1-\sigma(x))$
- Tanh: $\tanh(z)=1-\tanh^2(2)$
- ReLU: `ReLU(z) = 1 if z >= 0 else 0`
- Leaky ReLU: `leakyReLU(z) = 1 if z >= 1 else 0.01` 

# Gradient Descent for Neural Networks
- Assume we have one hidden layer 
- We have parameters:
    - $W^{[1]}$: size $(n^{[1]},n^{[0]})$ 
    - $b^{[1]}$: size $(n^{[0]}, 1)$
    - $W^{[2]}$: size $(n^{[2]},n^{[1]})$
    - $b^{[2]}$: size $(n^{[1]},1)$
- Our cost function is $J(W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]})=\frac1m\displaystyle\sum^n_{i=1}L(a^{[2]},Y)$
```
# Forward propagation
# Computing for hidden layer...
Z1 = W1 @ X + b1    # W1=(l,n); X=(n,m); b1=(l,1); then Z1=(l,m)
A1 = g1(Z1)         # A1=(l,m); g1 is our activation function, whether it be sigmoid, ReLU, etc.

# Computing for output layer
Z2 = W2 @ A1 + b2   # W2=(1,l); A1=(l,m); b2=(1,m); then Z2=(1,m)
A2 = g2(Z2)          # A2=(1,m)

# Backpropagation
dZ2 = (A2 - Y)                                  # A2=(1,m); Y=(1,m); then dZ2=(1,m); Assuming our activation function for output layer is sigmoid
dW2 = dZ2 @ A1.T / m                            # A1=(l,m); dZ2=(1,m); then dW2=(1,l)
db2 = np.sum(dZ2) / m   
dZ1 = W2.T @ dZ2 * derivg1(Z1)                  # W2=(1,l); dZ2=(1,m); derivg1(Z1)=(l,m); then dZ1=(l,m)
dW1 = dZ1 @ X.T / m                             # dZ1=(l,m); X=(n,m); dW1=(l,n)
db1 = np.sum(dZ1, axis=1, keepdims=True) / m    # dZ1=(l,m); db1=(l,1)    
```

# Random Initialization
- Initializing $W$ with 0's creates symmetry that we can't break, basically layers in node will output the same output as the others
- Instead initialize the weights as some random distribution with `np.random.randn(<shape>) * 0.01` 
    - The multiplier can be different, but usually pretty small

## Additional Notes
- The greater the hidden layer size, the better the accuracy (to a certain extent)
    - At a certain point it starts to overfit and accuracy starts decreasing