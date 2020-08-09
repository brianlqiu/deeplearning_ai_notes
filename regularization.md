# Regularization
- **Regularization** - a technique to reduce high variance
## Regularization with Logistic Regression
- $J(W,b)=\frac1mL(AL,Y)+\frac{\lambda}{2m}||W||^2_2$
    - Where $\lambda$ is the regularization parameter which is a hyperparameter set by the dev set
    - $||W||^2_2=\displaystyle\sum^{n_x}_{j=1}W_j^2=W^TW$ is $L_2$ regularization
- $L_1$ regularization is $||W||_1=\displaystyle\sum^{n_x}_{j=1}|W_j|$ 
    - Makes $W$ sparse (a lot more 0's) which can lead to less memory usage, but isn't as commonly used
## Regularization with Neural Networks
- $J(W,b)=\frac1m\displaystyle\sum^m_{i=1}L(A,Y)+\frac{\lambda}{2m}\displaystyle\sum^L_{l=1}||W^l||^2_F$
    - **Frobenius norm** = $||W^l||^2_F=\displaystyle\sum^{n^l}_{i=1}\displaystyle\sum^{n^{l-1}}_{j=1}(W^l_{i,j})^2$ = sum of all elements in $W^l$ squared 
- Whereas previously in backprop we would calculate $dW^l=dZ^l \bigotimes A^{l-1} $, with the regularization term we would calculate $dW^l=dZ^l \bigotimes A^{l-1}+\frac{\lambda}mW^l$, which is basically the derivative of the regularization term
- Essentially the regularization term scales the weight downwards by a factor of $(1-\alpha\frac{\lambda}m)$ when updating the parameters

# Regularization and Overfitting
- As $\lambda$ gets larger, the lambda decreases the impact of certain nodes
- If $\lambda$ is large, then $W$ will be smaller, which means that $Z$ and $A$ will also be much smaller, so that means our range of values for $A$ will be much smaller and more linear

# Dropout Regularization
- For each node, we set some probability of eliminating some nodes, train the model on that, then repeat and eliminate some other nodes, etc.
- Example: Assume we want to apply **inverted dropout** on layer 3
```
# Define some 'keep_prob' which is the probability that we keep that node
keep_prob = 0.8

# Let d3 be our dropout vector
d3 = np.random.randn(a3.shape[0], a3.shape[1]) > keep_prob
a3 *= d3            # 0 out all False values
# Since 20% of our nodes will get dropped out, this will scale Z downwards
# To account for that fact we have to divide by keep_prob to bump it back up
a3 /= keep_prob     
```
- At test time, we don't use dropout
- The inverted dropout step scales the output

## Why does dropout work?
- Over time, some units will overly rely on certain features, which can lead to overfitting
- Since a node cannot rely on certain input nodes/features with dropout, it spreads out the weight (which shrinks $W$)
- For larger layers, we can decrease `keep_prob` (to spread it out even more)
- For certain layers that shouldn't affect overfitting, we can keep `keep_prob = 1`
    - Usually input layer has `keep_prob = 1` or at least value very close to 1
- Only use dropout for overfitting models
- Disadvantage: Our cost function $J$ is no longer well-defined, since $J$ can be anything
    - First test a couple of iterations of gradient descent without dropout to confirm gradient descent is descending, then you can apply dropout

# Other Regularization Methods
- Data augmentation
    - If data is scarce, you can augment the data to produce more training examples
    - Example: With pictures, you can flip pictures horizontally, take random crops of images, make transformations, distortions, etc.
- Early stopping
    - Plot cost of gradient descent with respect to iterations, and also plot dev set error
    - Usually dev set error will go down, then start going back up at some point
    - Stop when dev set starts to go back up
    - Disadvantage: No longer fully optimizing the cost function
        - Try using L2 regularization if you have the time to adjust lambda