"""Cross-check hand-written metrics against scikit-learn / statsmodels-style references."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, brier_score_loss

from evaluation import metrics as M

rng = np.random.default_rng(0)


def test_confusion_scores_match_sklearn():
    labels = ['UP', 'FLAT', 'DOWN']
    y = rng.choice(labels, 2000); p = np.where(rng.random(2000) < 0.4, y, rng.choice(labels, 2000))
    r = M.confusion_and_scores(y, p, labels)
    assert abs(r['accuracy'] - accuracy_score(y, p)) < 1e-4
    assert abs(r['balanced_accuracy'] - balanced_accuracy_score(y, p)) < 1e-4
    assert abs(r['macro_f1'] - f1_score(y, p, average='macro')) < 1e-4
    assert abs(r['mcc'] - matthews_corrcoef(y, p)) < 1e-4


def test_probability_metrics_match_sklearn():
    y = rng.integers(0, 2, 3000); p = np.clip(0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.2, 3000), 0.01, 0.99)
    r = M.probability_report(p, y)
    assert abs(r['brier'] - brier_score_loss(y, p)) < 1e-4
    assert abs(r['roc_auc'] - roc_auc_score(y, p)) < 1e-3


def test_pesaran_timmermann_detects_skill_and_no_skill():
    actual = rng.random(5000) > 0.5
    _, p_noise = M.pesaran_timmermann(rng.random(5000) > 0.5, actual)
    skilled = np.where(rng.random(5000) < 0.6, actual, ~actual)
    _, p_skill = M.pesaran_timmermann(skilled, actual)
    assert p_noise > 0.01 and p_skill < 1e-6


def test_forecast_accuracy_random_walk_is_one():
    a = pd.Series(100 + rng.normal(0, 1, 500).cumsum())
    naive = a.shift(1)
    r = M.forecast_accuracy(a, naive, naive)
    assert abs(r['theils_u2'] - 1) < 1e-9 and abs(r['mase'] - 1) < 1e-9


def test_directional_report_oracle_is_perfect():
    ret = pd.Series(rng.normal(0, 0.01, 1000))
    r = M.directional_report(np.sign(ret), ret, 0.5, -0.5, pd.Series(0.001, index=ret.index), 1)
    assert r['hit_rate'] == 1.0
