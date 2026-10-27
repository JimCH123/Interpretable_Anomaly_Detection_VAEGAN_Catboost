"""CatBoost anomaly detector (dissertation Section IV-D)."""
from functools import partial

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from skopt import gp_minimize
from skopt.space import Integer, Real

from . import config


def load_dataset(path, target_col=config.TARGET_COL):
    """Load a combined dataset and return ``(data, X, y)`` with y in {0 normal, 1 anomaly}."""
    data = pd.read_csv(path)
    X = data.drop(columns=[target_col]).values
    y = (data[target_col].values == config.ANOMALY_LABEL).astype(int)
    return data, X, y


def split(X, y):
    return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.SEED)


def evaluate_detection(y_true, y_pred, y_scores):
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_scores),
        "pr_auc": average_precision_score(y_true, y_scores),
    }


# ----------------------------------------------------------------------------
# Bayesian hyperparameter search (Eq. 11)
# ----------------------------------------------------------------------------
PARAM_SPACE = [
    Integer(1, 5, name='max_depth'),
    Integer(10, 1000, name='iterations'),
    Real(0.001, 0.1, name='learning_rate', prior='log-uniform'),
    Integer(1, 20, name='l2_leaf_reg'),
    Integer(1, 40, name='min_data_in_leaf'),
]
PARAM_NAMES = [dim.name for dim in PARAM_SPACE]


def optimization(param_names, params, X_train, y_train):
    """Negative mean F1 over stratified K-fold CV (the objective minimised by ``gp_minimize``).

    The early-stopping set is split off the training fold; as in the original
    experiment, the model is fitted on the full training fold.
    """
    params_dict = dict(zip(param_names, params))
    s = config.BAYES_SEARCH

    kf = StratifiedKFold(n_splits=s["n_splits"], shuffle=True, random_state=config.SEED)
    f1s = []
    for train_idx, test_idx in kf.split(X_train, y_train):
        Xtrain, Xtest = X_train[train_idx], X_train[test_idx]
        ytrain, ytest = y_train[train_idx], y_train[test_idx]

        X_tr_sub, X_es, y_tr_sub, y_es = train_test_split(
            Xtrain, ytrain, test_size=0.2, random_state=config.SEED, stratify=ytrain)

        clf = CatBoostClassifier(**params_dict,
                                 loss_function='Logloss',
                                 eval_metric='F1',
                                 random_state=config.SEED,
                                 verbose=0)
        clf.fit(Xtrain, ytrain,
                eval_set=(X_es, y_es),
                early_stopping_rounds=s["early_stopping_rounds"],
                use_best_model=True)

        y_prob = clf.predict_proba(Xtest)[:, 1]
        y_pred = (y_prob >= config.DECISION_THRESHOLD).astype(int)
        f1s.append(f1_score(ytest, y_pred))

    return -1 * np.mean(f1s)


def bayesian_search_params(X_train, y_train):
    """Return the best ``{param: value}`` found and its cross-validated F1."""
    s = config.BAYES_SEARCH
    objective = partial(optimization, PARAM_NAMES, X_train=X_train, y_train=y_train)
    result = gp_minimize(
        objective,
        dimensions=PARAM_SPACE,
        n_calls=s["n_calls"],
        random_state=config.SEED,
        n_random_starts=s["n_random_starts"],
        verbose=True
    )

    best_params = {
        'max_depth': int(result.x[0]),
        'iterations': int(result.x[1]),
        'learning_rate': float(result.x[2]),
        'l2_leaf_reg': int(result.x[3]),
        'min_data_in_leaf': int(result.x[4])
    }
    best_f1 = -result.fun
    print(f"Best params: {best_params}, Best F1: {best_f1:.4f}")
    return best_params, best_f1


# ----------------------------------------------------------------------------
# Final model
# ----------------------------------------------------------------------------
def train_catboost(X_train, y_train, params=None):
    """Train the final detector.

    ``early_stopping_rounds`` has no effect here because no ``eval_set`` is
    passed, so the model always trains for the full number of iterations. This
    matches how the committed model was produced.
    """
    clf = CatBoostClassifier(**(params or config.CATBOOST_PARAMS))
    clf.fit(X_train, y_train, early_stopping_rounds=30)
    return clf


def predict(clf, X, threshold=config.DECISION_THRESHOLD):
    y_scores = clf.predict_proba(X)[:, 1]
    y_pred = (y_scores >= threshold).astype(int)
    return y_pred, y_scores


def save_model(clf, threshold=config.DECISION_THRESHOLD):
    config.MODEL.parent.mkdir(parents=True, exist_ok=True)
    clf.save_model(config.MODEL)
    np.save(config.THRESHOLD, threshold)
