import numpy as np

""" 1. Load the dataset """
# Assume our dataset is a set of (x,y) coordinates that are either red (y=0) or blue(y=1)
# X stores the points as a (2,m) matrix with 2 features (x and y) and m examples
# Y stores the correct label for each point in a (1,m) matrix with each column corresponding to an example
X, Y = load_planar_dataset()    

# Note: If the dataset is not linearly seperable, logistic regression does not perform well

""" 2. Define the neural network structure """
# We want our neural network to have 1 input layer with 2 nodes (1 for each coordinate), 1 hidden layer with 4 nodes,
# and one output layer with one node

def layer_sizes(X,Y):
    n_x = 2     # input layer
    n_h = 4     # hidden layer
    n_y = 1     # output layer
    return (n_x, n_h, n_y)

""" 3. Initialize the parameters """

def initialize_parameters(n_x, n_h, n_y):
    # Multiplier used so we don't start out at extreme values in our activation function, which will make slope close 
    # to 0 and thus gradient descent will be slow
    W1 = np.random.randn(n_h, n_x) * 0.01   
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2= np.zeros((n_y, 1))

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

""" 4. Forward propagate """

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = W1 @ X + b1    # W1=(4,2); X=(2,m); b1=(4,1); then Z1=(4,m)
    A1 = np.tanh(Z1)    # Assume we are using tanh activation function for our hidden layer
    Z2 = W2 @ A1 + b2   # W2=(1,4); A1=(4,m); b2=(1,1); then Z2=(1,m)
    A2 = sigmoid(Z2)    # Sigmoid is good choice for binary classification output

    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

""" 5. Compute cost """
def compute_cost(A2, Y, parameters):
    return np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / (-Y.shape[1])

""" 6. Backpropagate """
def tanh_derivative(Z):
    return 1 - (np.tanh(Z1)**2)

def backward_propagation(parameters, cache, X, Y):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]
    
    
    dZ2 = (A2 - Y) / m                      # A2=(1,m); Y=(1,m); dZ2=(1,m); Precomputed derivative                      
    dW2 = dZ2 @ A1.T                        # dZ2=(1,m); A1=(4,m); dW2=(1,4)
    db2 = np.sum(dZ2)                       # db2=(1,1)
    dZ1 = W2.T @ dZ2 * tanh_derivative(Z1)  # W2=(1,4); dZ2=(1,m); Z1=(4,m); dZ1=(4,m)
    dW1 = dZ1 @ X.T                         # dZ1=(4,m); X=(n,m); dW1=(4,n)
    db1 = np.sum(dZ1, axis=1, keepdims=True)# dZ1=(4,m); db1=(4,1)

    return {"dW2": dW2, "db2": db2, "dW1": dW1, "db1": db1}

""" 7. Update parameters """
def update_parameters(parameters, grads, learning_rate=1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

""" 8. Integrate into a model """
def nn_model(X, Y, n_h, num_iterations=10000):
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_iterations):
        cache = forward_propagation(X, parameters)
        cost = compute_cost(cache["A2"], Y, parameters)
        if i % 1000:
            print(f"Cost after iteration {i}: {cost}")
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)
    
    return parameters

""" 9. Predict """
def predict(parameters, X):
    cache = forward_propagation(X, parameters)
    predictions = cache["A2"] > 0.5

