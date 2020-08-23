# Orthogonalization
- **Orthogonalization** - making each tuning strategy only affect one aspect of the model

## Evaluation Metrics
- **Precision** - of the predictions made, what percent was correct
- **Recall** - what percent of a certain class was correcly predicted
    - Often there's a precision-recall tradeoff
    - Don't really want 2 metrics to judge our model on
- **$F_1$ score** - the Harmonic mean of precision and recall, given by the function:
    - $\frac2{\frac1P+\frac1R}$
- Lowest average error can also be a good choice of metric

### Satisficing and Optimizing Metric
- If you care about m metrics (say accuracy, running time, etc.) but one is more important than the others, then choose one metric (accuracy) to be your optimizing metric and the others to fall under some satisficing threshold (< 2000ms)

# Train/Dev/Test Distributions
- Make sure that dev and test sets come from the same distribution!
- Randomly shuffle data from the same distribution into dev & test sets
- Choose a dev and test set to reflect data you expect to get in the future and are important
## Ratios
- With smaller data sets, 70-30 or 60-20-20 split is a decent rule
- With larger data sets, 98-1-1 could be a good split
- The size of the test set should be big enough to give high confidence in the overall performance of your system
- You can skip having a test set, but a little more risky

# Changing Dev/Test Sets & Metrics
- If certain behaviors are undesirable that you haven't accounted for in evaluation metric, make sure to change the metric

# Human-level Performance
- Nowadays, models rapidly improve, surpassing human-level performance
- However, getting to the most optimal performance (**Baye's optimal error**) progresses very slowly and can never surpass
    - Once we get past human performance, there are techniques that can't be used or are more difficult to use for optimizing the model
        - Get labeled data from humans
        - Gain insight from manual error analysis
        - Analysis of bias/variance
- 2 goals:
    - Reduce bias/fit training set well until you near human-level performance/Baye's optimal error
        - Train bigger model
        - Train longer
        - Use optimization algorithms (momentum, RMSprop, Adam, etc.)
        - Find better NN architecture/hyperparameters
    - Reduce variance/predict well on the dev set/minimze difference between training set error and dev set error
        - More data
        - Regularization (L2, dropout, data agumentation, etc.)
        - Find better NN architecture/hyperparameters