# Binary Classification

## Classification with Cats

- Binary output: 0 if picture is not a cat; 1 if picture is a cat
- Images are stored in 3 different matrices corresponding to the red, green, and blue color channels
    - If an image is 64 x 64, then there a 3 64 x 64 matrices
- To turn all these matrices into feature vector, we create 1 column vector with R, G, B appended to each other
    - The dimensions are 1 x (64 * 64 * 3), call this dimension $n$ or $n_x$
- The feature vector will be the input

## Notation

- $m$ - number of training examples
    - $m_{train}$ - number of training examples
    - $m_{test}$ - number of test examples
- $n_x$ - dimension of the training example
- $x^i$ - the $i$th training example in vectorized form
- $y^i$ - the correct label for the $i$th training example 
- $X$ - $n_x$ x $m$ dimensional matrix containing all the vectorized training examples
- $Y$ - 1 x $m$ dimensional matrix containing correct labels for each training example

# Logistic Regression

- Logistic regression is an algorithm used for binary classification
- Given x, calculate the probability that $y=1$
    - Calculate $\hat y = P(y=1|x)$
- Parameters
    - $w$ - `$n_x$-dimensional vector
    - $b$ - some number $\in R$
- We want output to be some value between 0 or 1
- We can use a sigmoid function:
    - $\hat y = \sigma(w^Tx + b)$
    - Where the sigma function is defined as:
        - $\sigma(z) = \frac 1 {1 + e^{-z}}$
        - $\lim\limits_{z\rightarrow\infin}\sigma(z) = 1$
        - $\lim\limits_{z\rightarrow\ -\infin}\sigma(z) = 0$

# Logistic Regression Cost Function

- Let $x^i$ denote the $i$th training example
- Let $y^i$ denote the correct label for the $i$th training example
- Let $z^i = w^Tx^i+b$
- Given $\{(x^1, y^1), ...,(x^m, y^m)\}$, we want to find the closest approximate $\hat y^i \approx y^i$, where $\hat y^i = \sigma(z)$
- Possible loss functions to determine error for a **single** training example
    - Calculates how well your model predicts against the correct value
    - Difference squared function: $L(\hat y, y) = \frac 1 2 (\hat y - y)^2$
        - Not good for gradient descent
    - $L(\hat y, y) = -(y\log(\hat y) + (1 - y)\log(1 - \hat y))$
        - Better for gradient descent
        - If $y=1$, then $L(\hat y, y) = -\log(\hat y)$, if we want $\log(\hat y)$ to be large, then $\hat y$ must be large (as close to 1 as possible to get 0)
        - If $y=0$, then $L(\hat y, y) = -\log(1 - \hat y)$, if we want $\log(1 - \hat y)$ to be large, then $\hat y$ must be small (as close to 0 as possible to get 1)
- Cost function
    - Calculates loss over **all** training examples
    - $J(w, b) = \frac 1 m \displaystyle\sum_{i=1}^mL(\hat y^i, y^i)$

# Gradient Descent 

- Given the cost function $J(w,b)$, we want to find $w$ and $b$ such that $J(w,b)$ is minimized
    - Imagine the graph of $J$ on a 3D plane is a convex function, we want to find the minimize
- Gradient descent takes a step towards the direction with the greatest slope towards the minimum until it converges at the minimum
- Algorithm:
```
while True {
    w = w - alpha * dw  # where alpha = learning rate; dw = derivative of J with respect to w
    b = b - alpha * db  # where db = derivative of J with respect to b
}
```

# Computation Graphs

- Example: consider the function $J(a,b,c) = 3(a + bc)$
- The function can be decomposed into:
    - $u=bc$
    - $v=a+u$
    - $J=3v$
- This can be written as a "circuit" where the inputs are variables and the nodes are the operations (addition, multiplication, etc.) and the output is the output of $J(a,b,c)$

# Derivatives with Computation Graphs

```
# Forward Pass
a = 5
b = 3
c = 2

u = b * c
v = a + u
J = 3 * v

# Backwards Pass
dv = 3          # dJ/dv
da = dv * 1     # dJ/da = dJ/dv(3) * dv/da(1)
du = dv * 1     # dJ/du = dJ/dv(3) * dv/du(1)
db = du * c     # dJ/db = dJ/du(3) * du/db(c)
dc = du * b     # dJ/dc = dJ/du(3) * du/dc(b)
```

# Logistic Regression Gradient Descent
- $z = w^Tx + b$
- $\hat y=a=\sigma(z)$
- $L(a,y)=-(y\log(a)+(1-y)\log(1-a))$
- As an example, let our inputs be $x_1$, $w_1$, $x_2$, $w_2$, and $b$
    - Then $z=w_1x_1 + w_2x_2 + b$
    - Then $a=\sigma(z)$
    - Then we can calculate $L(a,y)$
- Derivative of loss function $L$:
    - We can assume $\log(a) = \ln(a)$
    - $\frac{dL}{da}=-(\frac ya+\frac{1-y}{1-a}*-1)=-\frac ya+\frac{1-y}{1-a}$
- Derivative of sigmoid function $a$:
    - $\sigma(z)=\frac1{1+e^{-z}}=(1+e^{-z})^{-1}$
    - $\frac{d\sigma}{dz}=-(1+e^{-z})^{-2}(-e^{-z})=\frac{e^{-z}}{(1+e^{-z})^2}=\frac{1+e^{-z}-1}{(1+e^{-z})^2}=\sigma(x)-\sigma(x)^2=\sigma(x)(1-\sigma(x))$
- Thus we can directly calculate $dL/dz$ as the following:
    - $\frac{dL}{dz}=\frac{dL}{da}\frac{da}{dz}=a-y$
```
# Forward pass, assume variables defined somewhere
z = w1 * x1 + w2 * x2 + b
a = 1 / (1 + e**(-1 * z))
L = -1 * (y * math.log10(a) + (1 - y) * Math.log(1 - a))

# Backwards pass
da = -((y / a) + -(1 - y) / (1 - a)) # derivative of loss function L, assume log is ln
dz = da * a * (1 - a)               # derivative of sigmoid function
dw1 = dz * x1                                 
dw2 = dz * x2
db = dz * 1

# Now apply gradient descent to adjust values
w1 = w1 - alpha * dw1
w2 = w2 - alpha * dw2
b = b - alpha * db
```

# Gradient Descent on m Examples
- Our cost function is $J(w,b)=\frac1m\displaystyle\sum_{i=1}^mL(a^i,y^i)$
    - Where $a^i=\sigma(z^i)=\sigma(w^Tx^i+b)$
    - We calculated gradient descent for $L$ in previous section
- To get gradient descent of $J$, $\frac{dJ}{dw_1}=\frac1m\displaystyle\sum_{i=1}^m\frac{dL(a^i,y^i)}{dw_1}$
- Algorithm:
```
# Given m examples and n features...
J = 0
dw = np.zeros(n, 1)         
db = 0

# Forward propagation
z = w.T @ X + b         # w.T is (1, n), X is (n, m), resulting vector is (1, m) with b broadcasted to each element
a = sigma(z)            # assume exists a function sigma that takes in a the (1, m) vector and returns (1, m) vector with sigmoid function applied
J = 1 / m * np.sum(a)   # calculates cost

# Backpropagation
dz = 1 / m * a - y      # use prederived trick in previous section, don't forget 1/m; a and y are both (1,m) so valid operation
dw = X @ dz.T           # have to transpose dz so we get (n,m) * (m,1) operation
db = np.sum(dz)                 

# Apply gradient descent
w = w - alpha * dw    
b = b - alpha * db         
```
- Vectorization is faster and avoids looping

# Vectorization
- GPU's and CPU's have parallelisation features (SIMD) that makes vectorization more efficient than looping 
- Try avoiding explicit for loops for neural networks

# Broadcasting
```
A = np.random.randn(3, 4)   

sumCols = A.sum(axis=0)         # get sum of all columns, get in (1,4) array
percentage = 100 * A / cal      # to calculate the percentages of each value in column on sum of column, we broadcast the (1,4) array cal to a (3,4), then divide all elements in A by their corresponding elements
```

# Learning Rates
- Dfiferent learning rates will determine how quickly the cost function converges
- The larger the learning rate, the bigger the "steps" gradient descent will take towards the minimum
- However, if the learning rate is too large, the algorithm may never converge
- If the learning rate is too low, there is a risk of overfitting