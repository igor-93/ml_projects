# OTTO project

[Kaggle Link](https://www.kaggle.com/c/otto-group-product-classification-challenge/overview/evaluation)

## Typical ML piepline:

- data analysis
    - check for missing data
    
- feature engineering
    - e.g. tech indicators at different scales for fin markets
    
- feature analysis
    - check correlation:
        - PCA: make orthogonal features
        - ICA: find independent compnents whose lin. comb. creates features
        - Kernel PCA
        
    - explained variance
        
    - feature importance
    
- labelling
    - define a meaningful label
    - regression might be more suitable since it can almost always we converted to binary 
    classification and it doesnt throw the infromation away
    - maybe multiple labels
    
- unbalanced data
    - sampling, bagging
    - take care about metrics DO NOT use accuracy (use e.g. precision, recall, etc..)
    
- modelling:
    - take a look into GLM, gradient boosting
    - maybe stack the models
        - model that picks models

