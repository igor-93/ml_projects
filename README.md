# ml_utils

Repository contains all ML-related code. 
Both functions commonly used in machine learning projects, 
as well as actual ML projects.

### Tests
From the `ml_utils/` directory run `python -m unittest discover`

TODO: add tests for KMedoids

### Useful packages:
   
- stacking and boosting of models: [vecstack](https://github.com/vecxoz/vecstack)
- model-based hyper-parameter search: [scikit-optimize](https://scikit-optimize.github.io/stable/)

# Typical ML pipeline:

- data analysis
    - check for missing data
    
- feature engineering
    - e.g. tech indicators at different scales for fin markets
    - [examples](https://www.kaggle.com/shahules/an-overview-of-encoding-techniques) 
    for categorical features
    
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