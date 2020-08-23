# Transfer Learning
- If you low data, freeze the earlier layers and just train the output layer
    - You can save the forward-prop data from the frozen layers to disk to avoid having to do forward-prop to the whole thing all over again
- With more data, still consider using transfer learning but freezing less layers
- Helps avoid tuning hyperparameters

# Data Augmentation
- Mirroring
- Random cropping
- Color shifting (change RGB values chosen from some random distribution)
    - PCA color augmentation - subtract more dominant color channels more than the less dominant to keep the overall tint

# Improving Benchmarks
- Ensembling - train several networks independently and average their outputs
- Mutli-crop at test time - run classifier on multiple versions of test images and average results