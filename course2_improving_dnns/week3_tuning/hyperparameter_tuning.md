# Hyperparameter Tuning
- Hyperapameters:
    - High priority:
        - $\alpha$
    - Medium priority:
        - $\beta$ for momentum
        - Hidden units
        - Mini-batch size
    - Low priority:
        - NUmber of layers
        - Learning rate decay
    - Negligible priority (almost never changed):
        - $\beta_1$
        - $\beta_2$
        - $\epsilon$
- Originally, we would try every combination of hyperparameters, which is comprehensive but computationally expensive
- Instead, randomly select some number of combinations of hyperparameters
- **Coarse to fine sweep** - if you scan over the grid of hyperparameters and see that some hyperparameters seem to perform best around some area, limit your search to that area and repeat the search

# Scaling Hyperparameter Search
- Uniform distribution isn't a great choice for all hyperparameters, we need to pick an appropriate scale
- Example: Say you want to tune $\alpha$ to some value between $0.0001$ and $1$, but you have some intuition that values on the lower end are better suited
    - You can use a log scale and pick a uniform sampling from that scale (i.e. 10 from range(0.0001, 0.001), 10 from range (0.001, 0.01), etc.)
- If you want to sample log scale between $10^a$ and $10^b$, select a number $r$ between $a$ and $b$ and set your hyperparameter to $10^r$

# Applying Hyperparameter Tuning
- Hyperparameters get stale; re-evaluate occasionally
- Two methods of tuning:
    - Babysitting one model
        - Usually done when there isn't a lot of computational power
        - After every time period (like 1 day), adjust hyperparameters
        - Can go back to previous day's model if a tuning goes wrong
    - Parallel training