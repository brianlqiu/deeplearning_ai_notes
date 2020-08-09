""" Mini-batch Gradient Descent """
# Build mini-batches from the training set #
def random_mini_batches(X, Y, mini_batch_size=64):
    # Shuffle the training set
    permutation = list(np.random.permutation(m))    # randomly permutes a list of 0 to m (exclusive)
    shuffled_X = X[:, permutation]                  # rearranges based on the randomly generated indices
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Partition the suffled training set
    num_complete_minibatches = m // mini_batch_size
    mini_batches = []
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[:, k * mini_batch_size : (k + 1) * mini_batch_size]    
        mini_batch_Y = Y[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # if there are leftover examples
    if m % mini_batch_size != 0:
        mini_batch_X = X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = Y[:, num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

""" Momentum """
# Initialize velocity to 0
def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)
    return v

# Update parameters using momentum
def update_parameters_with_momentum(parameters, grads, w, beta, learning_rate):
    L = parameters // 2

    for l in range(L):
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]
        parameters['W' + str(l + 1)] -= learning_rate * v["dW" + str(l + 1)]
        parameters['b' + str(l + 1)] -= learning_rate * v['db' + str(l + 1)] 

""" Adam Optimization """
def initialize_adam(parameters):
    v = {}
    s = {}
    L = len(parameters) // 2

    for l in range(L):
        v['dW' + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape)
        v['db' + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)
        s['dW' + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape)
        s['db' + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)

    return v, s

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2

    for l in range(L):
        v['dW' + str(l + 1)] = beta1 * v['dW' + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)] 
        v['db' + str(l + 1)] = beta1 * v['db' + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]
        vdW_corrected = v['dW' + str(l + 1)] / (1 - beta1**t)
        vdb_corrected = v['db' + str(l + 1)] / (1 - beta1**t)

        s['dW' + str(l + 1)] = beta2 * s['dW' + str(l + 1)] + (1 - beta2) * grads['dW' + str(l + 1)]**2
        s['db' + str(l + 1)] = beta2 * s['db' + str(l + 1)] + (1 - beta2) * grads['db' + str(l + 1)]**2
        sdW_corrected = s['dW' + str(l + 1)] / (1 - beta2**t)
        sdb_corrected = s['db' + str(l + 1)] / (1 - beta2**t)

        parameters['W' + str(l + 1)] -= learning_rate * (vdW_corrected / (np.sqrt(sdW_corrected) + epsilon))
        parameters['b' + str(l + 1)] -= learning_rate * (vdb_corrected / (np.sqrt(vdb_corrected) + epsilon))

