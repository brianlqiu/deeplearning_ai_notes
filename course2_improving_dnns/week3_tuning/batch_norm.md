# Batch Normalization
- For each layer, normalize the previous layer's output (this layer's input) 
- Given some intermediate values in a layer $z^{(1)},...,z^{[m]}$
    - Calculate $\mu=\frac1m\displaystyle\sum_iz^{(i)}$
    - Calculate $\sigma^2=\frac1m\displaystyle\sum_i(z^{(i)}-\mu)^2$
    - For each $z^{(i)}_{norm}=\frac{z^{(i)}-\mu}{\sqrt{\smash[b]{\sigma^2+\epsilon}}}$
    - Then $\hat z^{(i)}=\gamma z^{(i)}_{norm}+\beta$
        - Where $\gamma$ and $\beta$ are learnable parameters of the model
        - $\gamma$ allows you to choose the stddev
        - $\beta$ allows you to choose the mean
- Since we are adding a constant $\beta$, we don't need another constant $b$, so just remove $b$ completely
- Algorithm:
```
# Forward propagation
# Assume we are on layer L
Z[L] = W[L] @ A[L - 1]
mu[L] = np.sum(Z[L]) / m
sigmasqr = np.sum((Z[L] - mu)**2) / m
Znorm[L] = (Z[L] - mu) / (sigmasqr + epsilon)
Ztilde[L] = gamma[L] * Znorm[L] + beta[L]
A[L] = activation_function(Ztilde[L])

# Backpropagation
dZtilde[L] = dA[L] * activation_derivative(Ztilde[L])
dgamma[L] = dZtilde[L] @ Znorm[L]
dbeta[L] = np.sum(dZtilde[L], axis=1, keepdims=True)
dZnorm[L] = dZtilde[L] * gamma[L]
dZ[L] = dZnorm[L] / np.sqrt(sigmasqr + epsilon)
dW[L] = dZ[L] @ A[L - 1]
dA[L - 1] = W[L] @ dZ[L]
```
- Works with momentum, RMSprop, etc.

## Why does batch norm work?
- It makes weights on further more robust to changes made in layer 1
- **Covariate shift** - if you train your model on some distribution and then test that model on a model that differs, you need to retrain
    - Example: if you train model on only black cats and you try predicting on orange cats, you're probably not going to have a high accuracy
- In a normal neural network, in deeper layers, since all the layer knows is its previous input, the input will be changing all the time, leading to problems of covariate shift if the input values spreads out too much
- Batch normalization guarantees the inputs will have a certain mean/stddev
- Has a regularization effect
    - Each mini-batch is scaled by the mean/variance computed on that mini-batch
    - Adds noise to the $z$ values within that minibatch, which introduces noise (like dropout)

## Batch norm at test time
- Note that in forward propagation, $\mu$ and $\sigma^2$ are calculated with the sum of all examples
    - In test time, we might only have 1
- Use a $\mu$ and $\sigma^2$ equal to the the exponentially weighted average across mini-batches