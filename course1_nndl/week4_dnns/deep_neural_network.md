# Deep L-layer Neural Network
- Deep neural networks are neural networks with more layers
## Notation
- $L$ - the number of layers
- The input layer is referred to as layer 0
- $n^{[l]}$ - the number of nodes in the $l$th layer
- $a^{[l]}$ - the activation vector for the $l$th layer
    - $a^{[L]}$ - the prediction (activation vector for output)

# Forward Propagation in a Deep Network
- For each layer:
- $Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]}$
- $A^{[l]}=g^{[l]}(Z^{[l]})$

# General Dimensional Rules
- $W^{[l]}=(n^{[l]},n^{[l-1]})$
- $b^{[l]}=(n^{[l]}, 1)$

# Why Deep Representations? 
- Each node calculates smaller parts
- Each layer calculates certain parts
- The combination of layers composes the many outputs into a signular output
- Ex: For facial recognition, the first layer may have nodes that detect edges; the second layer can take those edges and compose them into body parts like eyes, eyebrows, lips, etc.; the third layer can start piecing together these indivudal parts; and the last layer can do a comparison of whole faces
- Ex: For real time speech-to-text, the first layer can be focused on detecting low-level audio waveforms; the second layer can translate these into phenomes; the third layer translates the phenomes into words; and the last layer transforms the words into sentences/phrases
## Circuit Theory
- There are functions that you can compute with a "small" L-layer deep neural network that shallower networks require exponentially more hidden units to compute

# Forward And Backward Propagation
## Forward Propagation
- Each layer takes in $A^{[l-1]}$ and outputs $A^{[l]}, cache[Z^{[l]}]$
- For each layer:
- $Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]}$
- $A^{[l]}=g^{[l]}(Z^{[l]})$
## Backward Propagation
- Each layer takes in $dA^{[l]}$ and outputs $dA^{[l-1]}, dW^{[l]}, db^{[l]}$
- For each layer:
- $dZ^{[l]}=dA^{[l]} * g'^{[l]}(Z^{[l]})$
- $dW^{[l]}=dZ^{[l]} \bigotimes A^{[l-1]}$
- $db^{[l]}=dZ^{[l]}$
- $dA^{[l-1]}=W^{[l]}dZ^{[l]}$

# Parameters vs Hyperparameters
- Our parameters are our $W^{[l]}$ and $b^{[l]}$
- **Hyperparameters** - parameters that affect the parameters
- Many different hyperparameters:
    - Learning rate $\alpha$
    - Iterations
    - Number of hidden layers
    - Number of nodes in each hidden layer
    - Choice of activation function
    - Momentum
    - Mini-batch size
    - Regularization
    - etc.
- Hyperparameters need to be tuned empirically
- At different periods of time, hyperparameters can be further optimized
    - Ex: Better hardware in the future might necessitate hyperparameter tuning for that hardware, etc.