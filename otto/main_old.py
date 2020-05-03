from os.path import join
from operator import itemgetter

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

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


def main(data_dir="data/"):
    file_train = join(data_dir, "train.csv")
    file_test = join(data_dir, "test.csv")
    df_train = pd.read_csv(file_train, index_col=0)
    if main_params["debug"]:
        df_train = df_train.iloc[np.random.randint(0, df_train.shape[0], 20000)]
    # df_test = pd.read_csv(file_test, index_col=0)
    print("loaded data")

    feat_cols = [c for c in df_train.columns if "feat" in c]
    df_train["target"] = df_train["target"].str.get(-1).astype(int)

    df_train["nz_feats"] = (df_train[feat_cols] != 0).sum(axis=1)
    df_train["sum_feats"] = df_train[feat_cols].sum(axis=1)

    df_train["n_1_5_feats"] = ((df_train[feat_cols] > 1) & (df_train[feat_cols] < 5)).sum(axis=1)
    df_train["n_5_10_feats"] = ((df_train[feat_cols] > 5) & (df_train[feat_cols] < 10)).sum(axis=1)
    df_train["n_10_20_feats"] = ((df_train[feat_cols] > 10) & (df_train[feat_cols] < 20)).sum(axis=1)

    X = np.array(df_train.drop(columns=["target"]).values)
    y = np.array(df_train["target"].values)
    cv = StratifiedKFold(main_params["cv"], shuffle=True)

    X = np.log(X + 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    meta_features = []

    print("#################### RUNNING MLP base models ####################")
    preds41 = run_rf(X, y, cv, tsne_or_sfc=None, rf_or_et="mlp")
    preds42 = run_rf(X, y, cv, tsne_or_sfc="sfc", rf_or_et="mlp")

    meta_features.extend([preds41, preds42])

    print("#################### RUNNING XGB base models ####################")
    preds01 = run_rf(X, y, cv, tsne_or_sfc=None, rf_or_et="xgb")
    preds02 = run_rf(X, y, cv, tsne_or_sfc="sfc", rf_or_et="xgb")

    meta_features.extend([preds01, preds02])

    print("#################### RUNNING RF base models ####################")
    preds11 = run_rf(X, y, cv, tsne_or_sfc=None, rf_or_et="rf")
    preds12 = run_rf(X, y, cv, tsne_or_sfc="sfc", rf_or_et="rf")

    meta_features.extend([preds11, preds12])

    print("#################### RUNNING ET base models ####################")
    preds13 = run_rf(X, y, cv, tsne_or_sfc=None, rf_or_et="et")
    preds14 = run_rf(X, y, cv, tsne_or_sfc="sfc", rf_or_et="et")

    meta_features.extend([preds13, preds14])

    print("#################### RUNNING k-NN base models ####################")
    preds21 = run_rf(X, y, cv, tsne_or_sfc=None, rf_or_et="knn")
    preds22 = run_rf(X, y, cv, tsne_or_sfc="sfc", rf_or_et="knn")

    meta_features.extend([preds21, preds22])

    print("#################### RUNNING T-SNE base models ####################")
    tsne = TSNE(n_components=3, perplexity=10)
    X = tsne.fit_transform(X)

    preds31 = run_rf(X, y, cv, tsne_or_sfc=None, rf_or_et="xgb", max_feats=3)
    preds32 = run_rf(X, y, cv, tsne_or_sfc=None, rf_or_et="rf", max_feats=3)
    preds33 = run_rf(X, y, cv, tsne_or_sfc=None, rf_or_et="knn", max_feats=3)

    meta_features.extend([preds31, preds32, preds33])

    meta_features = np.hstack(meta_features)

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
