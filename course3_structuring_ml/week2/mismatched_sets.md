# Training & Testing on Different Distributions
- Nowadays, many companies use a training set and testing set from different distributions
- Example: Say you have crawled high quality images and images from a user-uploaded dataset, and you want the model to perform well on the user-uploaded dataset
    - You could combine the two datasets, then do a 60-20-20 split; however, the high quality images are disproportionally represented in the test set and what you want to recognize
    - Better option: train with all high quality and some low quality; make dev & test all low quality

## Bias and Variance
- If high variance and training & dev set come from different distributions, a high variance could come from just general bad variance or the two different distributions are too far apart
- **Training-dev set** - same distribution as training set, but not used for training
- Now you can compare train error, training-dev error, and dev-error
    - If train error & training-dev error are far, that's just general high variance
    - If train error & training-dev error are close but dev set is high, then there's a data mismatch problem (distributions are too different)

# Addressing Data Mismatch
- Carry out manual error analysis to understand difference between training & dev sets
- Make training data more similar
- Collect more data

## Artificial Data Synthesis
- Using artificial means to create more data similar to the dev set
    - i.e. if your dev set is usually noisy and your training doesn't really have any noise, you can artifically add noise to the training data
- Be careful that your model might overfit to artifical additions if you reuse the same addition (i.e. keep using the same driving noise)