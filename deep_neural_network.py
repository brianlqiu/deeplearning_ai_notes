import numpy as np

""" 1. Initialize parameters """
# layer_dims - list of integers containing the number of nodes in each layer
def init_params(layer_dims):
    parameters = {}
    L = len(layer_dims) # Number of layers

    #  We start at 1 since we don't need weights for first layer (input)
    for l in range(1, L):   
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    
    return parameters

""" 2. Forward propagation """
# The linear step
def linear_forward(A, W, b):
    cache = (A, W, b)
    return (W @ A) + b, cache

# The activation function step
# Our two possible activation functions:
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z)), Z
def relu(Z):
    return np.maximum(Z, 0), Z

# activation - a string 'sigmoid' or 'relu' representing which activation function to use
def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = sigmoid(Z) if activation == 'sigmoid' else relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

""" 3. Apply forward propagation for each layer """
def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2    # We need to do divide by 2 to get # of params since each layer has a W & b

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches.append(cache)

    return AL, caches     

""" 4. Compute cost """
def compute_cost(AL, Y):
    return np.sum(Y * np.log(AL) + (1 - Y) * (np.log(1 - AL))) / -Y.shape[1]

""" 5. Backpropagation """
# cache - tuple (A_prev, W, b) from the current layer
def linear_backward(dZ, cache):
    # Our linear function is Z = WA + b
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = dZ @ A_prev.T                      
    db = np.sum(dZ, axis=1, keepdims=True)  
    dA_prev = W.T @ dZ

    return dA_prev, dW, db

# Derivative of sigmoid
def sigmoid_backward(dA, activation_cache):
    return dA * (sigmoid(Z) * (1 - sigmoid(Z)))

# Derivative of ReLU
def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.zeros(dA.shape)
    # Z <= 0 returns a boolean matrix of elements in Z that fulfill that condition, then numpy broadcasts it so that 
    # the setting applies to all indices
    dZ[Z <= 0] = 0          
    return dZ

# cache - tuple containing ((A, W, b), Z)
def linear_activation_backward(dA, cache, activation):
    return linear_backward(sigmoid_backward(dA, cache[1]) if activation == 'sigmoid' else relu_backward(dA, cache[1]), cache[0]) 

""" 6. Apply backprop for each layer """
def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    
    dAL = ((-Y / AL) + (1 - Y) / (1 - AL)) / m

    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = linear_activation_backward(grads["dA" + str(L + 1)], current_cache, 'relu')
    
    return grads
    
""" 7. Update parameters """
def update_params(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
    
    return parameters

""" 8. Integrate parts into a model """
def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000):
    costs = []
    parameters = init_params(layers_dims)

    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        coss.append(cost)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_params(parameters, grads, learning_rate)
    
    return parameters

""" 9. Predict """
def predict(X, Y, parameters):
    AL, caches = L_model_forward(X, parameters)
    predictions = np.zeros(AL.shape)
    predictions[AL > 0.5] = 1
    predictions[AL <= 0.5] = 0
    return predictions
