from os.path import join, isdir
from operator import itemgetter
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
import lightgbm as lgb

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from xgboost import XGBClassifier

from otto.nn_model import MyNN

main_params = {
    "cv": 5,  # 10
    "n_hyper_runs": 20,  # 100
    "n_hyper_starts": 10,  # 10
    "n_best_from_base": 2,
    "debug": False
}


def confusion_matrix_df(y_true, y_pred, n_classes=9):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = [cm[i, i] / cm[:, i].sum() for i in range(cm.shape[0])]

    cm = np.vstack([cm, precision])
    cm = pd.DataFrame(cm, index=[f"true_{i + 1}" for i in np.arange(n_classes)] + ["precision"],
                      columns=[f"pred_{i + 1}" for i in np.arange(n_classes)])
    cm["recall"] = [cm.iloc[i, i] / cm.iloc[i, :].sum() for i in range(n_classes)] + [accuracy]

    return cm


def load_data(data_dir):
    assert isdir(data_dir)
    file_train = join(data_dir, "train.csv")
    file_test = join(data_dir, "test.csv")
    df_train = pd.read_csv(file_train, index_col=0)
    df_test = pd.read_csv(file_test, index_col=0)

    df_train["target"] = df_train["target"].str.get(-1).astype(int)
    print(f"Data shape: {df_train.shape}")

    # shuffle
    df_train = df_train.sample(frac=1, random_state=442)
    return df_train, df_test


def pre_process(df_train: pd.DataFrame, df_test: pd.DataFrame, drop_feats=True):
    n, d = df_train.shape
    d = d - 1 # remove target dim
    feat_cols = [c for c in df_train.columns if "feat" in c]

    # add new features
    d = d + 1
    df_train[f"feat_{d}"] = (df_train[feat_cols] != 0).sum(axis=1)
    df_test[f"feat_{d}"] = (df_test[feat_cols] != 0).sum(axis=1)
    d = d + 1
    df_train[f"feat_{d}"] = df_train[feat_cols].sum(axis=1)
    df_test[f"feat_{d}"] = df_test[feat_cols].sum(axis=1)

    to_drop_before_pca = ['feat_46', 'feat_13', 'feat_44', 'feat_93', 'feat_63', 'feat_74',
                          'feat_81', 'feat_51', 'feat_12', 'feat_6', 'feat_31', 'feat_87',
                          'feat_5', 'feat_65', 'feat_73']

    # split into X, y
    if drop_feats:
        df_clean = df_train.drop(columns=to_drop_before_pca)
        df_clean_te = df_test.drop(columns=to_drop_before_pca)
    else:
        df_clean = df_train
        df_clean_te = df_test
    X = df_clean.drop(columns="target").astype(dtype=float)
    y = df_clean["target"] - 1
    d = X.shape[1]

    train_size = int(0.92 * X.shape[0]) # 0.92
    valdiation_id = int(0.99 * X.shape[0]) # 0.99

    # take log because data is log-normal
    X = np.log(X + 1)
    X_test_all = np.log(df_clean_te + 1)

    # split into train, validation, test
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size].values.astype("int64")
    X_val, y_val = X.iloc[train_size:valdiation_id], y.iloc[train_size:valdiation_id].values.astype("int64")
    X_test, y_test = X.iloc[valdiation_id:], y.iloc[valdiation_id:].values.astype("int64")
    print(f"train size      : {X_train.shape}")
    print(f"test size       : {X_test.shape}")
    print(f"validation size : {X_val.shape}")

    # apply scaler and PCA
    pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=d))])
    X_train_np = pipe.fit_transform(X_train).astype("float32")
    X_test_np = pipe.transform(X_test).astype("float32")
    X_val_np = pipe.transform(X_val).astype("float32")
    X_test_all_np = pipe.transform(X_test_all).astype("float32")

    X_train = pd.DataFrame(X_train_np, index=X_train.index)
    X_val = pd.DataFrame(X_val_np, index=X_val.index)
    X_test = pd.DataFrame(X_test_np, index=X_test.index)
    X_test_all = pd.DataFrame(X_test_all_np, index=X_test_all.index)

    return X_train, y_train, X_val, y_val, X_test, y_test, X_test_all


def run_base_model(X, y, X_val, y_val, X_test_all, cv, model, n_classes):
    # Note: test set only to be used for showing confusion matrix, NOT for hyper param tuning
    # either fix the hyper params, of grid-search them with cv,
    # actually better to fix them here and grid-search them in the notebook

    eval_set = [(X_val, y_val)]

    y_train_pred = np.zeros((X.shape[0], n_classes), dtype=float)
    y_test_pred = []
    for testset in X_test_all:
        y_test_pred.append(np.zeros((testset.shape[0], n_classes), dtype=float))
    if model in ["lgbm", "nn"]:
        if model == "lgbm":
            params = {
                "learning_rate": 0.01,
                "num_leaves": 500,
                "n_estimators": 1500,
                "max_depth": 25,
                "min_data_in_leaf": 30,
                "subsample": 0.4,
                "bagging_freq": 1,
                "feature_fraction": 0.6,
                "early_stopping_rounds": 10,
            }
            clf = lgb.LGBMClassifier(silent=False, objective="softmax", num_class=n_classes, verbose=0, n_jobs=8,
                                     **params)
        elif model == "nn":
            clf = MyNN(n_classes=n_classes)
        else:
            raise ValueError(model)
        print("Fitting model on each of the CV splits...")
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print("  fitting...")
            clf.fit(X_train, y_train, eval_set=eval_set)
            y_train_pred[test_index, :] = clf.predict_proba(X_test)

        print("Fitting model on the whole train data...")
        clf.fit(X, y, eval_set=eval_set)
        for i in range(len(X_test_all)):
            y_test_pred[i][:, :] = clf.predict_proba(X_test_all[i])

    elif model in ["rf", "knn", "et"]:
        if model == "rf":
            clf = RandomForestClassifier(n_jobs=-1, n_estimators=1000,
                                         max_depth=32, max_features=0.6, min_samples_leaf=5, verbose=1)
        elif model == "knn":
            k = 125
            clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        elif model == "et":
            clf = ExtraTreesClassifier(n_estimators=1000,
                                       n_jobs=-1, max_depth=32, max_features=0.7, max_samples=0.6,
                                       min_samples_leaf=1, verbose=1)
        else:
            raise ValueError(f"Unknown model: {model}")

        print("Fitting model on each of the CV splits...")
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print("  fitting...")
            clf.fit(X_train, y_train)
            y_train_pred[test_index, :] = clf.predict_proba(X_test)

        print("Fitting model on the whole train data...")
        clf.fit(X, y)
        for i in range(len(X_test_all)):
            y_test_pred[i][:, :] = clf.predict_proba(X_test_all[i])

    else:
        raise ValueError(f"Unknown model: {model}")

    # print confusion matrix for each of the models
    cm = confusion_matrix_df(y_true=y, y_pred=y_train_pred.argmax(axis=1))
    print(cm)
    return y_train_pred, y_test_pred


def tune(X_train: np.array, y_train: np.array, X_val, y_val, X_test, y_test, n_classes, cv):
    # params = {
    #     "learning_rate": 0.01,
    #     "num_leaves": 500,
    #     "n_estimators": 1500,
    #     "max_depth": 25,
    #     "min_data_in_leaf": 30,
    #     "subsample": 0.4,
    #     "bagging_freq": 1,
    #     "feature_fraction": 0.6,
    #     "early_stopping_rounds": 10,
    # }
    # clf = lgb.LGBMClassifier(silent=True, objective="softmax", num_class=n_classes, verbose=0, **params)

    eval_set = [(X_val.values, y_val)]

    assert (y_train < n_classes).all()
    assert 0 in y_train
    assert (y_test < n_classes).all()
    assert 0 in y_test

    clf = MyNN(n_classes=n_classes)
    clf.fit(X_train.values, y_train, eval_set=eval_set)

    y_pred = clf.predict_proba(X_test.values)
    y_pred = np.argmax(y_pred, axis=1)
    cm = confusion_matrix_df(y_pred=y_pred, y_true=y_test)
    print(cm)

    # param_grid = {
    #     "n_neighbors": [7, 11]
    # }
    #
    # clf = KNeighborsClassifier(n_jobs=4)
    #
    # grid_search = GridSearchCV(clf, n_jobs=2, param_grid=param_grid, cv=cv, scoring="neg_log_loss", verbose=2)
    # grid_search.fit(X_train, y_train)
    # print(f"Score: {grid_search.best_score_}")
    #
    # print(f"Best params: {grid_search.best_params_}")

    sys.exit(0)


def main(data_dir="data/"):
    df_train, df_test = load_data(data_dir=data_dir)
    n_classes = 9
    X_train, y_train, X_val, y_val, X_test, y_test, X_test_sub = pre_process(df_train, df_test)

    # use the data as following:
    # 1. train: for training AND hyper-param tuning
    # 2. validation: for early stopping
    # 3. test: for final evaluation

    cv = StratifiedKFold(5, shuffle=True, random_state=442)

    # tune(X_train, y_train, X_val, y_val, X_test, y_test, n_classes=n_classes, cv=cv)

    all_meta_feats_tr = []
    all_meta_feats_te = []
    all_meta_feats_sub = []

    models = ["nn", "knn"] # ["lgbm", "knn", "rf", "nn"]
    feature_names = []
    for model in models:
        print(f"#################### RUNNING {model} base model ####################")
        meta_feats_tr, meta_feats_te = run_base_model(X=X_train.values, y=y_train,
                                                      X_val=X_val.values, y_val=y_val,
                                                      X_test_all=[X_test.values, X_test_sub.values],
                                                      n_classes=n_classes,
                                                      cv=cv, model=model)
        all_meta_feats_tr.append(meta_feats_tr)
        all_meta_feats_te.append(meta_feats_te[0])
        all_meta_feats_sub.append(meta_feats_te[1])
        feature_names.extend([f"{model}_{i+1}" for i in range(meta_feats_tr.shape[1])])

    all_meta_feats_tr = np.hstack(all_meta_feats_tr)
    all_meta_feats_te = np.hstack(all_meta_feats_te)
    all_meta_feats_sub = np.hstack(all_meta_feats_sub)
    # convert to DF and save for further analysis
    all_meta_feats_tr = pd.DataFrame(all_meta_feats_tr, columns=feature_names, index=X_train.index)
    all_meta_feats_tr["y_true"] = y_train
    all_meta_feats_te = pd.DataFrame(all_meta_feats_te, columns=feature_names, index=X_test.index)
    all_meta_feats_te["y_true"] = y_test
    all_meta_feats_sub = pd.DataFrame(all_meta_feats_sub, columns=feature_names, index=X_test_sub.index)
    all_meta_feats_tr.to_csv("meta_features_train_fixed_nn.csv")
    all_meta_feats_te.to_csv("meta_features_test_fixed_nn.csv")
    all_meta_feats_sub.to_csv("meta_features_submission_fixed_nn.csv")
    print(f"Saved train meta features. Dim: {all_meta_feats_tr.shape}")
    print(f"Saved test meta features. Dim: {all_meta_feats_te.shape}")
    print(f"Saved submission meta features. Dim: {all_meta_feats_sub.shape}")

    all_meta_feats_tr.drop(columns=["y_true"], inplace=True)
    all_meta_feats_te.drop(columns=["y_true"], inplace=True)

    # train 2nd level model on the meta features
    scaler = StandardScaler()
    all_meta_feats_tr = scaler.fit_transform(all_meta_feats_tr)
    all_meta_feats_te = scaler.transform(all_meta_feats_te)
    all_meta_feats_sub = scaler.transform(all_meta_feats_sub)

    params = {
        "learning_rate": 0.01,
        "num_leaves": 8,
        "n_estimators": 800,
        "max_depth": 25,
        "min_data_in_leaf": 30,
        "subsample": 0.8,
        "bagging_freq": 5,
        "feature_fraction": 0.4,
    }
    clf = lgb.LGBMClassifier(silent=False, objective="softmax", num_class=n_classes, verbose=0, **params)

    clf.fit(all_meta_feats_tr, y_train)
    y_pred_tr = clf.predict_proba(all_meta_feats_tr)
    y_pred_te = clf.predict_proba(all_meta_feats_te)
    y_pred_sub = clf.predict_proba(all_meta_feats_sub)

    cm_tr = confusion_matrix_df(y_true=y_train, y_pred=y_pred_tr.argmax(axis=1))
    cm_te = confusion_matrix_df(y_true=y_test, y_pred=y_pred_te.argmax(axis=1))

    print("Confusion matrix train:")
    print(cm_tr)

    print("Confusion matrix test:")
    print(cm_te)

    y_pred_sub = pd.DataFrame(y_pred_sub, index=X_test_sub.index,
                              columns=[f"Class_{i}" for i in range(1,n_classes+1)])
    y_pred_sub.to_csv("submission.csv")


if __name__ == '__main__':
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    main(join(dir_path, "data"))
