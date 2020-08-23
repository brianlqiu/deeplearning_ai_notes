# Transfer Learning
- If you have a neural network that is already learned image recognition, you can change the output layer to fit the dataset you want to recognize and train those weights
    - You can only train the last layer, or provided you have enough data, you could retrain the whole model
    - If you retrain, this is called pre-training
    - If you retrain only last layer, this is called fine-tuning
- Good if you don't have a lot of new data
- The two tasks have to take the same input
- Works better if the two tasks do similar tasks on low level features

# Multi-task Learning
- If in one example, you need to identify multiple labels (i.e. in a picture, if there's a car, stop sign, etc.), you should have an output layer equal to the number of labels
- Cost function is now changed to $\frac1m\displaystyle\sum^m_{i=1}\displaystyle\sum^C_{j=1}L(A^i_j, Y^i_j)$, where $i$ is the number of examples and $C$ is the number of labels
- Works with a dataset with a subset of the labels (not all training examples need all the labels)
## When to use multi-task learning
- When training on a set of tasks that share lower-level features
- When the amount of data you have for each task is quite similar
- When you can train a big enough neural network to do well on all the tasks
    - It can hurt if your neural network is too small