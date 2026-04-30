"""Random-Forest classifier wrapper for status labels."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .labeler import LABEL_ORDER


@dataclass
class ClassifierEval:
    accuracy: float
    confusion: np.ndarray
    label_order: list[str]
    classification_report_str: str


def train_classifier(
    X: np.ndarray, y: list[str],
    n_estimators: int = 300,
    max_depth: int | None = None,
    test_size: float = 0.2,
    seed: int = 7,
    n_jobs: int = -1,
):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=None
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=n_jobs,
    )
    clf.fit(X_tr, y_tr)
    return clf, (X_tr, X_te, y_tr, y_te)


def evaluate_classifier(clf, X_te: np.ndarray, y_te: list[str],
                        labels: list[str] = LABEL_ORDER) -> ClassifierEval:
    y_pred = clf.predict(X_te)
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    acc = float(np.mean(np.array(y_pred) == np.array(y_te)))
    rep = classification_report(y_te, y_pred, labels=labels, zero_division=0)
    return ClassifierEval(
        accuracy=acc,
        confusion=cm,
        label_order=labels,
        classification_report_str=rep,
    )


def classifier_uncertainty(clf, X: np.ndarray) -> np.ndarray:
    """Per-sample 1 − max(predict_proba). Higher → more uncertain."""
    p = clf.predict_proba(X)
    return 1.0 - p.max(axis=1)
