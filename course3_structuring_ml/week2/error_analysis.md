# Error Analysis
- Example: Suppose your cat classifier has 90% accuracy and 10% error
    - It seems like some of the mislabeled images were dogs
    - To determine whether you should spend your time training your model to avoid dogs, add error analysis
        - Get around 100 mislabeled dev set examples
        - Get the percentage of the mislabeled examples were dogs
        - To calculate the upper bound of accuracy you can get by resolving this problem, multiply the percentage of mislabeled examples that were dogs to the overall error and subtract it with the error
            - i.e. if 5% of mislabeled images in your sample were dogs, by training the dog example you will probably get around $100(0.1-(0.1*0.05))=9.5\%$ error
        - Depending on how large this decrease is, you can decide whether or not to spend your time training
- Go through a sampling of the mislabeled dev set manually to see what's causing errors and carry out error analysis on the different categories

# Cleaning Up Incorrectly Labeled Data
- DL algorithms are quite robust to random errors in the training set, as long as there aren't too many errors
- Systematic errors are bad
- For cleaning up dev test, when carrying out error analysis add incorrectly labeled category of error
    - First look at overall dev set error; if low, then ok; if not continue
    - Compare incorrect label category percentage to other error category percentages; if low, then ok; if high, then consider cleaning up
    - If the difference between 2 models on the dev set is within the margin of error of the mislabeled images percentage, then you should probably clean up the dev test so you get better data
- If you do clean up your dev set, make sure to clean up the test set as well
- Consider examining examples your algorithm got right that should've been wrong (since mislabeled)