# Mini-batch Gradient Descent
- With normal gradient descent, we have to process the entire training set for each iteration of gradient descent, which is computationally expensive for large datasets
- Instead, separate the training set into mini-batches of some size smaller than the training set
    - Let $X^{\{i\}}$ denote the $i$th mini-batch of training examples
    - Let $Y^{\{i\}}$ denote the $i$th mini-batch of training example labels
    - Let $T$ denote the number of mini-batches
- Algorithm:
```
for t in range(1, T + 1):
    AL, caches = forward_propagate(X[t], parameters, b)
    cost = compute_cost(AL, Y[t])
    grads = back_propagate(caches)
    parameters = update_parameters(parameters, grads, learning_rate)
```
- Each pass through a mini-batch is called an epoch
- If you plot cost for normal gradient descent, cost should decrease for every iteration
- If you plot cost for mini-batch gradient descent, cost doesn't necessarily decrease on every iteration, since you are training on different training sets in each iteration, but it should trend downwards
## Mini-batch Sizes
- **Batch gradient descent** - if the size of mini-batch is m (the whole set)
    - Good optimization, but computationally expensive
- **Stochastic gradient descent** - if the size of mini-batch is 1 (each example is a single example)
    - Won't ever converge, since individual examples can shift the weights in totally different directions
    - Random direction not that big of a deal, but you lose the speed from vectorization
- **Mini-batch gradient descent** - some size in between m and 1
    - Takes advantage of vectorization and can also make progress without going through the whole set (best of both, although to a lesser degree)
## General Rules for Choosing Mini-batch Sizes
- If $m \leq 2000$: use batch gradient descent
- Typical mini-batch sizes: powers of 2 up to 512
    - Powers of 2 can be more efficient due to computer memory addressing
- Make sure mini-batches fit in CPU/GPU memory

# Exponentially Weighted Averages
- $V_t=\beta v_{t-1} + (1-\beta)\theta_t, \beta=0.9$
- $V_t$ is an approximate average over $\frac1{1-\Beta}$ days temperature
    - So if $\beta=0.9$, then we are computing over the last 10 days average 
- The greater the value of $\beta$, the more smooth and more latency is produced, since we are averaging over more days
- The smaller the value of $\beta$, the graph produced trends more closely to the data
- Not as accurate as keeping a moving window of days (using a queue or something like that), but faster and less memory
## Bias Correction
- Since we initialize v to 0, we start off much lower than what it should be, since the current data is still being weighted by 0
- Instead of taking $V_t$, use $\frac{V_t}{1-\beta^t}$
    - Since as t gets large, $\beta^t$ approaches 0 (since it's < 1), which means that as we move on this bias correction affects the output less

# Gradient Descent with Momentum
- When taking steps in gradient descent, we want to have a faster learning when moving directly towards the minimum and slower learning when moving indirectly towards the minimum
    - Since if we move indirectly in too big of a step, we can overshoot and diverge
- With momentum, on iteration $t$:
    - Compute $dW$ and $dB$ on the current mini-batch
    - Calculate the exponentially weighted average of $dw$: $V_{dW}=\beta V_{dW} + (1-\beta)dW$
    - Calculate the weighted average for $db$ as well
    - Update $W$ and $b$ using $V_{dW}$ and $V_{db}$ instead of $dW$ and $db$
- Smooths out the oscillations in indirect steps
    - Since we are oscillating, the steps in those directions average out to around 0, while our direction towards the minimum is preserved since we are still moving towards the minimum
- Momentum analogy
    - Imagine the function trying to optimize is a bowl, and you're trying to roll a ball towards the bottom
    - $dW$ and $db$ are acceleration
    - $V_{dw}$ and $V_{db}$ can represent velocity
    - $\beta$ represents friction
    - As the ball rolls towards the bottom, we generate momentum
- Most commonly chosen value of $\beta=0.9$
- Bias correction not commonly used, but can still be used
- In some literature, the calculation is represented as $V_{dw}=\beta V_{dW} + dW$, and the omitted $1-\beta$ is accounted for in the learning rate $\alpha$
- Will work better than gradient descent almost always

# RMSprop
- RMSprop stands for root mean square prop
- Same problem as momentum; want to move fast directly towards the minimum; want to move slower indirectly towards the minimum (perpendicular to the direction towards the minimum)
    - Let $W$ be the direction directly towards the minimum
    - Let $b$ be the direction perpendicular to $W$
- To implement RMSprop, on iteration $t$:
    - Compute $dW$ and $db$ on current mini-batch
    - $S_{dW}=\beta_2 S_{dW} + (1-\beta)dW^2$
    - $S_{db}=\beta_2 S_{db} + (1-\beta)db^2$
    - $W=W-\alpha\frac{dW}{\sqrt{\smash[b]{S_{dW}}} + \epsilon}$
    - $b=b-\alpha\frac{db}{\sqrt{\smash[b]{S_{db}}} + \epsilon}$
- If we are taking large steps in the $b$ direction, then $S_{db}$ will be relatively large, making updates in the $b$ direction much smaller (we are dividing)
- If we are taking smaller steps in the $W$ direction, then $S_{dW}$ will be relatively small, making updates in the $W$ direction much larger
- We need to add $\epsilon$ to the denominator since there's a possibility that the terms can be very small and we don't want it to blow up
- Can use larger learning rates since we probably won't diverge
- Note: $W$ and $b$ are not actually the parameters that affect in which direction we take a step in: usually there is a set of parameters that lead us to take indirect steps and a set of parameters that leads us to take direct steps 

# Adam Optimization
- A combination of RMSprop and momentum
- Algorithm:
    - Initialize $V_{dW}=0$, $S_{dW}=0$, $V_{db}=0$, $S_{db}=0$
    - On iteration $t$:
        - Compute $dW$ and $db$ on current mini-batch
        - The momentum step: 
            - Calculate $V_{dW}=\beta_1V_{dW}+(1-\beta_1)dW$
            - ...
        - The RMSprop step:
            - Calculate $S_{dW}=\beta_2S_{dW}+(1-\beta_2)dW^2$
            - ...
        - Bias correct $V_{dW}$, $V_{db}$, $S_{dW}$, and $S_{db}$:
            - $V_{dW} = \frac{V_{dW}}{1-\beta_1^t}$
            - ...
        - Update parameters $W$ and $b$
            - $W=W-\alpha\frac{V_{dW}}{\sqrt{\smash[b]{S_{dW}}} + \epsilon}$
            - ...
- Hyperparameters:
    - $\alpha=?$ - needs to be tuned
    - $\beta_1=0.9$
    - $\beta_2=0.999$
    - $\epsilon=10^{-8}$
    - Usually default values of $\beta_1$, $\beta_2$, and $\epsilon$ are used
- Adam: Adaptive moment estimation

# Learning Rate Decay
- Slowly reducing learning rate over time
- At beginning of training, set learning rate large because not really a risk of diverging
- When nearing the minimum, decreasing learning rate so we don't overshoot & diverge
- An epoch is one pass through a mini-batch
- $\alpha=\frac1{1+rn}\alpha_0$
    - Where $r$ is the decay rate
    - Where $n$ is the epoch number
- You can tune the initial learning rate $a_0$ and the decay rate $r$
- Other formulas:
    - $\alpha=0.95^n*\alpha_0$, where $n$ is the epoch number (exponential decay)
    - $\alpha=\frac k{\sqrt{\smash[b]{n}}}$, where $n$ is the epoch number
    - Discrete formula: after some number of epochs, decrease it by half, etc.
    - Manually decaying

# Local Optima
- Imagine a plane with many hills and valleys
    - People worried that we would reach the bottom of one of these local optima and get stuck
- Now we know that at points of local optima where the slope is 0, these are usually saddle points, so we can still reach the minimum
## Plateaus
- If the surface is relatively flat, it can take a long time to get the minimum of these plateaus and then go down
- Adam usually helps fix this problem