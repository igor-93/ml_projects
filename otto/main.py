from os.path import join
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

from otto.sparse_feature_compressor import SparseFeatureCompressor

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


def run_rf(X, y, cv, tsne_or_sfc=None, rf_or_et="rf", max_feats=10):
    n_classes = len(np.unique(y))
    n_hyper_runs = main_params["n_hyper_runs"]
    n_hyper_starts = main_params["n_hyper_starts"]

    space = [
        Integer(1, 10, name='rf__max_depth'),
        Integer(2, max_feats, name='rf__max_features'),
        Integer(1, 50, name='rf__min_samples_leaf'),
        Real(0.2, 0.6, "uniform", name='rf__max_samples')
    ]

    n_estimators = 1000
    if rf_or_et == "rf":
        clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
    elif rf_or_et == "et":
        clf = ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=-1)
    elif rf_or_et == "xgb":
        clf = XGBClassifier(n_estimators=n_estimators, n_jobs=-1)
        space = [
            Integer(1, 10, name='rf__max_depth'),
            Integer(1, 50, name='rf__min_samples_leaf'),
            Real(10 ** -5, 10 ** 0, "log-uniform", name='rf__learning_rate'),
            Real(1e-3, 1, "log-uniform", name='rf__reg_alpha'),
            Real(1e-3, 1, "log-uniform", name='rf__reg_lambda'),
            Real(0.2, 0.6, "uniform", name='rf__subsample')
        ]
    elif rf_or_et == "knn":
        space = [
            Integer(3, 200, name='rf__n_neighbors'),
            Categorical(["uniform", "distance"], name='rf__weights'),
            Integer(1, 2, name='rf__p'),
        ]
        clf = KNeighborsClassifier(n_jobs=-1)
    elif rf_or_et == "mlp":
        clf = MLPClassifier(learning_rate="adaptive", verbose=False, early_stopping=True)
        space = [
            Categorical([(256, 256), (256, ), (128, ), (32, 64, 32)], name='rf__hidden_layer_sizes'),
            Real(1e-5, 10, "log-uniform", name='rf__alpha'),
        ]
        n_hyper_runs = 8
        n_hyper_starts = 3
    else:
        raise ValueError()

    pipeline_list = [('rf', clf)]

    if tsne_or_sfc == "sfc":
        dim_red = SparseFeatureCompressor()
        pipeline_list = [('dim_red', dim_red)] + pipeline_list

        space.append(Integer(5, 30, name="dim_red__n_components"))
    elif tsne_or_sfc is None:
        pass
    else:
        raise ValueError()

    pipe = Pipeline(pipeline_list)

    @use_named_args(space)
    def objective_old(**params):
        pipe.set_params(**params)
        y_pred = []
        y_true = []
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            pipe.fit(X_train, y_train)
            y_pred.append(pipe.predict_proba(X_test))
            y_true.append(y_test)

        y_pred = np.vstack(y_pred)
        y_true = np.hstack(y_true)
        assert len(y_pred.shape) == 2, f"y_pred: {y_pred.shape}"
        assert len(y_true.shape) == 1, f"y_true: {y_true.shape}"
        assert y_pred.shape[1] == n_classes
        # assert np.array_equal(clf.classes_, np.arange(1, 10)), f"classes_ is not sorted: {rf.classes_}"

        loss = log_loss(y_true=y_true, y_pred=y_pred)
        return loss

    @use_named_args(space)
    def objective(**params):
        pipe.set_params(**params)
        return -np.mean(cross_val_score(pipe, X, y, cv=cv, n_jobs=-1, scoring="neg_log_loss"))

    print(f"Searching for hyper-params: n_hyper_runs={n_hyper_runs}")
    res_gp = gp_minimize(objective, space, n_calls=n_hyper_runs,
                         n_random_starts=n_hyper_starts, random_state=0, verbose=True)

    best_named_params = {s.name: p for s, p in zip(space, res_gp.x)}
    print(f"""RF Best parameters: {best_named_params}""")

    print(f"Best score={res_gp.fun}")
    take_best = main_params["n_best_from_base"]
    print("res_gp.func_vals ", res_gp.func_vals)
    idx = np.argpartition(res_gp.func_vals, take_best, axis=0)[:take_best]
    # k_best_params = np.array(res_gp.x_iters)[idx]
    k_best_params = itemgetter(*idx)(res_gp.x_iters)
    print("k_best_params ", k_best_params)

    # for each set of top k params, train a model on CV - 1 folds and then predict the other set
    level1_predictions = []
    for params_i in range(take_best):
        params = k_best_params[params_i]
        print("params: ", params)
        named_params = {s.name: p for s, p in zip(space, params)}
        pipe.set_params(**named_params)

        y_pred = np.zeros((X.shape[0], n_classes), dtype=float)
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            pipe.fit(X_train, y_train)
            y_pred[test_index, :] = pipe.predict_proba(X_test)

        print("y_pred sum: ", (y_pred.sum(axis=1) == 1.0).all())

        level1_predictions.append(y_pred)

    level1_predictions = np.hstack(level1_predictions)
    assert level1_predictions.shape[1] == take_best * n_classes, f"level1_predictions: {level1_predictions.shape}"
    print("####################")
    return level1_predictions


def load_data(data_dir):
    assert os.path.isdir(data_dir)
    file_train = os.path.join(data_dir, "train.csv")
    # file_test = os.path.join(data_dir, "test.csv")
    df_train = pd.read_csv(file_train, index_col=0)
    # df_test = pd.read_csv(file_test, index_col=0)

    df_train["target"] = df_train["target"].str.get(-1).astype(int)
    print(f"Data shape: {df_train.shape}")

    # shuffle
    df_train = df_train.sample(frac=1, random_state=442)
    return df_train


def pre_process(df_train: pd.DataFrame):
    n, d = df_train.shape
    d = d - 1 # remove target dim
    feat_cols = [c for c in df_train.columns if "feat" in c]

    # add new features
    d = d + 1
    df_train[f"feat_{d}"] = (df_train[feat_cols] != 0).sum(axis=1)
    d = d + 1
    df_train[f"feat_{d}"] = df_train[feat_cols].sum(axis=1)

    to_drop_before_pca = ['feat_46', 'feat_13', 'feat_44', 'feat_93', 'feat_63', 'feat_74',
                          'feat_81', 'feat_51', 'feat_12', 'feat_6', 'feat_31', 'feat_87',
                          'feat_5', 'feat_65', 'feat_73']

    # split into X, y
    df_clean = df_train.drop(columns=to_drop_before_pca)
    X = df_clean.drop(columns="target")
    y = df_clean["target"] - 1
    d = X.shape[1]

    train_size = int(0.7 * X.shape[0])
    valdiation_id = int(0.8 * X.shape[0])
    print(f"train_size: {train_size}, d: {d}")

    # take log because data is log-normal
    X = np.log(X + 1)

    # split into train, validation, test
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_val, y_val = X.iloc[train_size:valdiation_id], y.iloc[train_size:valdiation_id]
    X_test, y_test = X.iloc[valdiation_id:], y.iloc[valdiation_id:]

    # apply scaler and PCA
    pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=d))])
    X_train = pipe.fit_transform(X_train)
    X_test = pipe.transform(X_test)
    X_val = pipe.transform(X_val)

    return X_train, y_train, X_val, y_val, X_test, y_test


def run_base_model(X, y, X_val, y_val, cv, model, n_classes=9):
    # Note: test set only to be used for showing confusion matrix, NOT for hyper param tuning
    # either fix the hyper params, of grid-search them with cv,
    # actually better to fix them here and grid-search them in the notebook

    y_pred = np.zeros((X.shape[0], n_classes), dtype=float)
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
        lg = lgb.LGBMClassifier(silent=False, objective="softmax", num_class=n_classes, verbose=1, **params)
        eval_set = [(X_val, y_val)]

        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            lg.fit(X_train, y_train, eval_set=eval_set)
            y_pred[test_index, :] = lg.predict_proba(X_test)

    elif model in ["rf", "knn"]:
        # TODO: set parameters
        if model == "rf":
            clf = RandomForestClassifier(n_estimators=1500, n_jobs=-1)
        elif model == "knn":
            k = 5  # TODO
            clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model: {model}")

        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            y_pred[test_index, :] = clf.predict_proba(X_test)

    else:
        raise ValueError(f"Unknown model: {model}")

    # print confusion matrix for each of the models
    cm = confusion_matrix_df(y_true=y, y_pred=y_pred.argmax(axis=1))
    print(cm)
    return y_pred


def tune(X_train, y_train, cv):
    param_grid = {
        "max_depth": [8, 16, 32],
        "min_samples_leaf": [5, 10, 30],
        "max_features": [0.4, 0.6],
    }

    rf = RandomForestClassifier(n_estimators=700, n_jobs=1)

    grid_search = GridSearchCV(rf, n_jobs=-1, param_grid=param_grid, cv=cv, scoring="neg_log_loss", verbose=2)
    grid_search.fit(X_train, y_train)
    print(f"Score: {grid_search.best_score_}")

    print(f"Best params: {grid_search.best_params_}")
    sys.exit(0)


def main(data_dir="data/"):
    df_train = load_data(data_dir=data_dir)
    X_train, y_train, X_val, y_val, X_test, y_test = pre_process(df_train)

    # use the data as following:
    # 1. train: for training AND hyper-param tuning
    # 2. validation: for early stopping
    # 3. test: for final evaluation

    cv = StratifiedKFold(10, shuffle=True, random_state=442)

    tune(X_train, y_train, cv=cv)

    meta_features = []

    models = ["mlp", "lgbm", "rf", "et", "knn", "svm"]
    feature_names = []
    for model in models:
        print(f"#################### RUNNING {model} base model ####################")
        meta_features = run_base_model(X=X_train, y=y_train, X_val=X_val, y_val=y_val, cv=cv, model=model)
        meta_features.append(meta_features)
        feature_names.extend([f"{model}_{i+1}" for i in range(meta_features.shape[1])])


    meta_features = np.hstack(meta_features)
    # convert to DF and save for further analysis



    # train 2nd level model on the meta features
    scaler = StandardScaler()
    meta_features = scaler.fit_transform(meta_features)

    print(f"Num features at level 2: {meta_features.shape[1]}")

    clf = XGBClassifier(n_estimators=1000, max_depth=8, subsample=0.4, n_jobs=-1)
    final_scores = cross_val_score(clf, meta_features, y, scoring="neg_log_loss", cv=5, n_jobs=-1, verbose=1)

    print(f"Level 2 scores: \n{final_scores}")


if __name__ == '__main__':
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    main(join(dir_path, "data"))
