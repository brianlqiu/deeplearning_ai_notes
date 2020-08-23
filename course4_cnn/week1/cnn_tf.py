import numpy as np
import tensorflow as tf

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name='X')
    Y = tf.placeholder(tf.float32, [None, n_y], name='Y')

    return X, Y

def initialize_parameters(m, n_H, n_W, n_C, name):
    W = tf.get_variable(name, [m,n_H,n_W,n_C], initializer=tf.contrib.layers.xavier_initializer())
    return W

def forward_propagation(X, parameters, stride_conv, stride_pool, pool_filters, num_output):
    L = len(parameters)

    P = X
    for l in range(L):
        # We don't need to stride over examples or channels 
        Z = tf.nn.conv2d(P, parameters['W' + str(l + 1)], 
                        strides=[1, stride_conv['s' + str(l + 1)], stride_conv['s' + str(l + 1)], 1], 
                        padding='SAME') # Conv step
        A = tf.nn.relu(Z)   # ReLU activation step
        # Pool based on filter size for this layer
        P = tf.nn.max_pool(A, ksize=[1,pool_filters['f' + str(l + 1)],pool_filters['f' + str(l + 1)],1], 
                        strides=[1, stride_pool['s' + str(l + 1)], stride_pool['s' + str(l + 1)], 1], 
                        padding='SAME') 
    
    F = tf.contrib.layers.flatten(P)    # returns (m,k) matrix, where k = h*w*c
    Z = tf.contrib.layers.fully_connected(F,num_output,activation_fn=None)

def compute_cost(Z, Y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009, num_epochs=100, minibatch_size=64):
    (m, n_H0, n_W0, n_C0) = X_train.shape 
    X, Y = create_placeholders(n_H0, n_W0, n_C0, m)
    parameters = initialize_parameters()
    Z = forward_propagation(X, parameters)
    cost = compute_cost(Z, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            minibatch_cost = 0.0
            num_minibatches = m // minibatch_size
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:
                (mminibatch_X, minibatch_Y) = minibatch

                _, temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches

    return parameters