# Setting Up Training/Dev/Test Sets
- Applied ML is an iterative process for tuning hyperparameters
    - Layers
    - Hidden units
    - Learning rates
    - Activation functions
    - etc.
- Split data into different sets:
    - **Training set** - set of data used for training
    - **Hold out cross validation/development set** - data used for tuning/validating against training set
    - **Test set** - final test set to get a view of your model performance
- Size of each set depends on the size of your data
    - Dev and test sets don't need to be that large, dev set only needs to be large enough to differentiate between different algorithms
    - For smaller datasets, 60-20-20 or 70-30 could be good ratios
    - For larger datasets, since dev & test sets don't need to be that large, you could go for 98-1-1 or 99.5-0.25-0.25 or 99.5-0.4-0.1
## Mismatched Train/Test Distribution
- Training set can come from a different distribution than dev & test sets, but dev & test sets have to come from the same distribution
- Ex: Training set composed of very high-resolution pictures; dev & test set composed of user-uploaded varying-resolution pictures
- Not having a test set might be OK

# Bias & Variance
- **High bias/Underfitting** - model does not fit the data
    - If high training set error and high dev set error are approximately the same, indicates underfitting (bad performance on both)
- **High variance/Overfitting** - model perfectly fits the data used to train, but doesn't hold for general cases  
    - If low training set error and high dev set error, indicates overfitting
- If high training set error and much higher dev set error, indicates both under and overfitting
- Baye's error helps set the standard of which to compare error to

# Basic Recipe
- Does the model have high bias (poor training set performance)? If so, try the following util bias is low:
    - Bigger network
    - Longer training time
    - Other neural network architectures
- Does the model have high variance (poor dev set performance)? If so, try the following and train again until variance is low:
    - More data
    - Regularization
    - Other neural network architectures