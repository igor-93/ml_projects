import pandas as pd


def near_zero_variance_analysis(X):
    zero_var_df = pd.DataFrame(False, index=X.columns,
                               columns=['freqRatio', 'percentUnique', 'zeroVar', 'nearZeroVar'])
    zero_var_df = zero_var_df.astype(
        dtype={'freqRatio': "float32", "percentUnique": "float32", "zeroVar": "bool",
               "nearZeroVar": "bool"})
    for c in X.columns:
        feature = X[c]
        val_counts = feature.value_counts()
        if len(val_counts) == 1:
            zero_var_df.loc[c, 'zeroVar'] = True
        perc_of_uniq = float(len(val_counts)) / float(len(feature))
        freq_ratio = val_counts.iloc[0] / val_counts.iloc[1]
        zero_var_df.loc[c, 'freqRatio'] = freq_ratio
        zero_var_df.loc[c, 'percentUnique'] = perc_of_uniq
        if perc_of_uniq < 0.2 and freq_ratio > 20:
            # print(val_counts)
            zero_var_df.loc[c, 'nearZeroVar'] = True

    return zero_var_df


def draw_umap(X, y, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    import umap
    import matplotlib.pyplot as plt
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(X);
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)), c=y)
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1], c=y)
    if n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], c=y, s=100)
    plt.title(title, fontsize=18)
