# OTTO project

[Kaggle Link](https://www.kaggle.com/c/otto-group-product-classification-challenge/overview/evaluation)

## Current model
Stacking.
Steps:
1. Remove some features (According to feature importance done in analysis)
1. Take log of features since thy are counts so that standard scaler makes sense
1. PCA without dim reduction to remove correlations
1. Train base models: random forest, k-NN, neural net, light-GBM
    - also tried extreme trees but removed them because of too high 
      correlation with random forest
    - for all of the models, I first ran grid-search to find best params and
      then used the same CV folds to generate meta-features (to avoid leakage)
1. Train 2nd-level model on meta features + base features
    - I initially tried just with meta features, but adding base features improved 
      the result by a bit
    - didn't really run grid-search on the 2nd-level, but should have


## Future work
1. Improve the 2nd level model. So far I just tried one set of
parameters with 5-fold CV that was just used for early stopping. Ofc also increase n folds.
2. Maybe NN model on the 2nd level?
3. Smarter loss function for one of the base models to give more weight 
for common miss-classification. E.g. when miss-classifying 3rd and 4th class as 2rd 
we should increase the loss by e.g. 10%
