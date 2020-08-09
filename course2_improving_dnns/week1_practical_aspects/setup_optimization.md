# Normalizing Inputs
- When normalizing our inputs, we are centering and distributing the inputs evenly
```
mean = np.sum(X) / m
stddev = np.sum(X**2) / m
x -= mean
x /= stddev
```
## Why normalize?
- If inputs are on much different scales, we have to use a much smaller learning rate since the risk of overshooting is much higher
    - Visualize the graph

# Vanishing/Exploding Gradients
- At certain points in gradient descents, gradients can get extremely small or large
- If weights are large and neural network is deep, the weights can grow exponentially
- If weights are small and neural network is deep, the weights can decrease exponentially

## A Parital Solution: Weight Initialization
- For a single node's weights $w_i$, we want $Var(w_i)=\frac1n$
    - For ReLU functions, He initalization ($\frac2n$) is usually a better choice
- To do so, initialize the weights to the following (for ReLU):
`W[l] = np.random.randn(<shape>) * np.sqrt(2 / n[l-1])`
- For $\tanh$, Xavier initialization $\frac1{n^{l-1}}$ is usually better
- $\frac2{n^{l-1}+n^l}$ can also be a good choice
- Doesn't solve the problem of exploding/vanishing gradients, but can help

# Gradient Checking
- Take all your weights $W$ and $b$ and reshape int a large vector $\theta$
- Take all the derivatives of your weights $dW$ and $db$ and reshape into a big vector of $d\theta$
- For each $\theta^i$
    - Calculate $d\theta_{approx}^i=\frac{J(\theta^1, ...\theta^i+\epsilon,...\theta^L)-J(\theta_1,...\theta^i-\epsilon,...\theta^L)}{2\epsilon}$, which gives you a vector of approximate gradients
    - Then calculate $\frac{||d\theta_{approx}-d\theta||_2}{||d\theta_{approx}||_2+||d\theta||_2}$ and compare against some value (anything larger than $10^{-5}$ is worrying)
    - Try using an epsilon value of $10^{-7}$
## Notes on Applying Gradient Checking
- Only use gradient checking when debugging (computationally expensive)
- If algorithm fails grad check, look at components and try to identify bug
- Remember the regularization term
- Gradient check doesn't work with dropout, make sure to turn off dropout before running
- Sometimes derivatives seem to work at certain values of weights and biases but become inaccurate at different weights; try checking these weights later on

