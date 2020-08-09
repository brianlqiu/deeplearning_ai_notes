import numpy as np  # Library for matrix operations
import h5py         # Library for loading files in h5 format
    
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")  # Open file for reading
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])    # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])    # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])       # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])       # your test set labels

    classes = np.array(test_dataset["list_classes"][:])             # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

""" 1. Loading the dataset """

# train_set_x_orig - (m,w,h,c) matrix containing m training examples (pictures) with a width w=64, height h=64, and 
#                     number of RGB channels c=3 
# train_set_y - (1,m) matrix containing the m correct labels (1 if cat, 0 if not) for train_set_x_orig
# test_set_x_orig - (m,w,h,c) matrix containing m test examples (pictures) with a width w=64, height h=64, and 
#                    number of RGB channels c=3 
# test_set_y - (1,m) matrix containing the m correct lables for test_set_x_orig
# classes - (2,1) matrix containing the 2 different classes for the examples (1 if cat, 0 if not)  
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

""" 2. Vectorizing the dataset """
# If X = (a,b,c,d), then X.reshape(X.shape[0], -1) = (a,b*c*d)

# Creates (w*h*c,m) matrix, where each row is a RGB channel value (0-255) and each column is a different picture 
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T  
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

""" 3. Standardizing the dataset """
# Normally standardization means subtracting the mean and dividing by stddev, but for picture datasets, dividing every 
# element by 255 (max value) seems to work well
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

""" 4. Initializing parameters """
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b

""" 5. Optimizing parameters """
# Defining our activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Calculates one iteration of gradient descent
def propagate(w, b, X, Y):
    ## Forward propagation ##
    # wX + b is the raw output that will be fed to the sigmoid function to see if it activates
    A = sigmoid(w.T @ X + b)

    # Our loss function is -(ylog(a) + (1-y)log(1-a))
    # If the correct label is 1 (y=1) and a<=0.5, then loss gets large, since the prediction is incorrect
    # If the correct label is 1 (y=1) and a>0.5, then loss gets small, since the prediction is correct
    # If the correct label is 0 (y=0) and a<=0.5, then loss gets small, since the prediction is correct
    # If the correct label is 0 (y=0) and a>0.5, then loss gets large, since the prediction is incorrect
    
    # Our cost is the average of the loss function calculated on all training examples
    cost = np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / -m
    
    ## Backward propagation ##
    # Gradient of cost with respect to w
    # dw = dcost * dloss * dsigmoid * (X) // X = derivative of z with respect to w
    # dw = (1/m) * (-y/a + (1-y)/(1-a)) * a(1 - a)
    # dw = (a - y)/m
    dw = X @ ((A - Y) / m).T    # Since X is (n,m) and A & Y are both (1,m), we need to transpose
    
    # Gradient of cost with respect to b
    # Same process as before, but since derivative of z with respect to b is just 1, we don't need to multiply anything
    db = np.sum((A - Y) / m)    

    grads = {"dw": dw, "db": db}
    
    return grads, cost

# Runs gradient descent repeatedly to optimize the weights and biases
def optimize(w, b, X, Y, num_iterations, learning_rate):
    costs = []
    for i in range(num_iterations):
        # Run iteration of gradient descent
        grads, cost = propagate(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]

        # Update weights
        w -= learning_rate * dw
        b -= learning_rate * db

        # Every 100 iterations get the cost
        if i % 100 == 0:
            costs.append(cost)

    params = {"w": w, "b": b}       # Get final parameters
    grads = {"dw": dw, "db": db}    # Get final gradients

    return params, grads, costs

""" 6. Making predictions with the optimized parameters """
def predict(w, b, X):
    A = sigmoid(w.T @ X + b)    # Apply the parameters on the set of images to be predicted
    
    Y_prediction = np.zeros((1, m))

    # Convert probabilities generated by sigmoid to 1 or 0
    for i in range(A.shape[1]):
        Y_prediction[0][i] = 1 if A[0][i] > 0.5 else 0  # If probability > 0.5, then output 1
    
    return Y_prediction

""" 7. Merging functions into a model """
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5):
    w, b = initialize_with_zeros(X_train.shape[0])  # initialize n weights

    # Get the optimized parameters by running gradient descent for the specified number of iterations
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

    w = parameters["w"]
    b = parameters["b"]

    # Try predicting for testing and training set
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

