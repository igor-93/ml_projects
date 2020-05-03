import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, log_loss
from sklearn.utils.class_weight import compute_sample_weight


def feat_importance_mdi(forest, feat_names):
    """
    Takes average and std of feature importance over all trees in the random forest.
    :param forest: TRAINED forest
    :param feat_names:
    :return:
    """
    df = {i: tree.feature_importances_ for i, tree in enumerate(forest)}
    df = pd.DataFrame.from_dict(df, orient='index')
    df.columns = feat_names
    df.replace(0, np.nan, inplace=True)
    importance = pd.concat({'mean': df.mean(), 'std': df.std() * df.shape[0]**(-0.5)}, axis=1)
    importance /= importance['mean'].sum()

    importance.sort_values(['mean'], axis=0, ascending=False, inplace=True)

    return importance


def feat_importance_mda(clf, X, y, cv, feat_names, scoring='f05'):
    """
    Mean Decrease Accuracy
    :param clf: classifier on which we will call fit and predict
    :param Xs: list of k X matrices. Each matrix is a CV subset of common
    :param ys: list of labes corresponding to XS
    :param feat_names: feature names
    :param scoring: type of scoring: either negative log loss or accuracy
    :return:
    """
    # feat importance based on OOS score reduction
    if len(feat_names) != X.shape[1]:
        raise ValueError('feat_names len is wrong: {} vs {}'.format(len(feat_names), Xs[0].shape[1]))
    if scoring not in ['neg_log_loss', 'accuracy', 'f1', 'f05']:
        raise ValueError('Wrong scoring method.')

    scr_before, scr_after = pd.Series(), pd.DataFrame(columns=feat_names)

    unique, class_counts = np.unique(y, return_counts=True)
    n_classes = len(unique)
    class_weights = y.shape[0] / (n_classes * class_counts)
    class_weights = dict(zip(unique, class_weights))
    print('class_weights: ', class_weights)

    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w_test = compute_sample_weight(class_weights, y=y_test)
        if issparse(X_test):
            X_test = X_test.todense()

        fit = clf.fit(X=X_train, y=y_train)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X_test)
            scr_before.loc[i] = -log_loss(y_test, prob, sample_weight=w_test, labels=clf.classes_)
        elif scoring == 'accuracy':
            pred = fit.predict(X_test)
            scr_before.loc[i] = accuracy_score(y_test, pred, sample_weight=w_test)
        elif scoring == 'f1':
            pred = fit.predict(X_test)
            scr_before.loc[i] = f1_score(y_test, pred, average='weighted')
        elif scoring == 'f05':
            pred = fit.predict(X_test)
            tmp = fbeta_score(y_test, pred, beta=0.5, average='weighted')
            print('tmp: ', tmp)
            scr_before.loc[i] = tmp

        print(f'Permuting {len(feat_names)} features: {i+1}/{cv.get_n_splits()}')
        for j, feat_name in enumerate(feat_names):
            X1_ = X_test.copy()
            np.random.shuffle(X1_[:, j])  # permutation of a single column
            if scoring == 'neg_log_loss':
                prob = fit.predict_proba(X1_)
                scr_after.loc[i, feat_name] = -log_loss(y_test, prob, sample_weight=w_test, labels=clf.classes_)
            elif scoring == 'accuracy':
                pred = fit.predict(X1_)
                scr_after.loc[i, feat_name] = accuracy_score(y_test, pred, sample_weight=w_test)
            elif scoring == 'f1':
                pred = fit.predict(X1_)
                scr_after.loc[i, feat_name] = f1_score(y_test, pred, average='weighted')
            elif scoring == 'f05':
                pred = fit.predict(X1_)
                tmp = fbeta_score(y_test, pred, beta=0.5, average='weighted')
                print('tmp: ', tmp)
                scr_after.loc[i, feat_name] = tmp

    imp = (-scr_after).add(scr_before, axis=0)
    if scoring == 'neg_log_loss':
        imp = imp / -scr_after
    else:
        imp = imp / (1. - scr_after)
    importance = pd.concat({'mean': imp.mean(), 'std': imp.std() * imp.shape[0] ** -.5}, axis=1)

    importance.sort_values(['mean'], axis=0, ascending=False, inplace=True)

    return importance, scr_before.mean()