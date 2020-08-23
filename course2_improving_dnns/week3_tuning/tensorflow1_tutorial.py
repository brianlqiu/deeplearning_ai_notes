import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

""" Tensorflow Steps """
# 1. Create Tensors (variables) that are not executed/evaluated
y_hat = tf.constant(36, name='y_hat')
y = tf.constant(39, name='y')

# 2. Write operations between those Tensors
loss = tf.Variable((y - y_hat)**2, name='loss') # assume our loss function is L=(yhat - y)^2

# 3. Initialize your Tensors
init = tf.global_variables_initializer()

# 4. Create a Session
with tf.Session() as session:
    # 5. Run your session
    session.run(init)
    print(session.run(loss))

""" Placeholders """
# A placeholder is a variable that can't have a value until it is fed in at run time with feed_dict
x = tf.placeholder(tf.int64, name='x')
print(sess.run(2 * x, feed_dict={x:3}))
sess.close()

""" Linear Function """
def linear_function():
    X = tf.constant(np.random.randn(3,1), name='X')
    W = tf.constant(np.random.randn(4,3), name='W')
    b = tf.constant(np.random.randn(4,1), name='b')
    Y = W @ x + b

    with tf.Session() as session:
        result = session.run(Y)

    return result

""" Sigmoid """
def sigmoid(Z):
    X = tf.placeholder(tf.float32, name='X')
    sigmoid = tf.sigmoid(Z)

    with tf.Session() as session:
        result = session.run(sigmoid, feed_dict={X: Z})
        
    return result

""" Cost """
def cost(logits, labels):
    # logits = A[L], labels = Y
    z = tf.placeholder(tf.float32, name='z')
    y = tf.placeholder(tf.float32, name='y')

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)

    with tf.Session() as session:
        cost = session.run(cost, feed_dict={z: logits, y: labels})

    return cost

""" One-hot Encoding """
# Given a vector Y with values ranging from 0 to C-1, where C = number of classes, this represents a matrix X where 
# if Y[i] = C, then X[i][C] = 1 and the rest of the column = 0
def one_hot_matrix(labels, C):
    C = tf.constant(C, name='C')
    one_hot_matrix = tf.one_hot(labels, C, axis=1)
    with tf.Session() as sess:
        one_hot = sess.run(one_hot_matrix)
    return one_hot

""" Ones """
def ones(shape):
    ones = tf.ones(shape=shape)
    with tf.Session() as sess:
        ones = sess.run(ones)

    return ones

""" Neural Network """
# 1. Load dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# 2. Flatten & normalize dataset
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

X_train = X_train_flatten/255.
X_test = X_test_flatten/255.

Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

# 3. Create placeholders
def create_placeholders(n_x, n_y):
    # Use none since we don't know how many examples we have
    return tf.placeholder(tf.float32, [n_x, None], name='X'), tf.placeholder(tf.float32, [n_y, None], name='Y') 

# 4. Initalize parameters
def initialize_parameters():
    W1 = tf.get_variable('W1', [25,12288], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', [25,1], initializer=tf.zeros_initializer())
    # ...

# 5. Forward propagation
def forward_propagation(X, parameters):
    # Get parameters...

    Z1 = W @ X + b1
    A1 = tf.nn.relu(Z1)
    # ... Forward propagation stops at Z3, since when we're training we don't need A3!

# 6. Compute cost
def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)   # Tensorflow requires rows=m, cols=classes
    labels = tf.transpose(Y)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

# 6a & b. Backprop & parameter updates
# Handled by Tensorflow!

# 7. Build the model
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost = True):
    ops.reset_default_graph()   # to be able to rerun model w/o overwriting variables
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X, Y = create_placeholders(n_x, n_y)
    parameters = initailize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(X, Z3)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # does the backprop for us
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = m // minibatch_size
            minibatces = random_mini_batches(X_train, Y_train, minibatch_size)  # Defined in previous programs

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                # use _ variable if we don't need it
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / minibatch_size

                costs.append(epoch_cost)

        plt.plot(np.squeeze(costs)) 
        plt.ylabel('cost')
        plt.xlabel('iterations (x5)')
        plt.title('learning rate=' + str(learning_rate))
        plt.show()

        # Save parameters
        parameters = sess.run(parameters)

        # Calculate correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))