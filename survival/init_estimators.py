from survival.selectors import fit_and_score_features, ManualSelector, SelectKBestCustom
from sklearn.feature_selection import SelectKBest

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
import numpy as np

def init_model(seed, n_workers, model_name):
    models_dict = {
        'CoxPH': CoxPHSurvivalAnalysis(n_iter=100000),
        'RSF': RandomSurvivalForest(random_state=seed, n_jobs=n_workers, n_estimators=50, max_depth=None),
    }
    return models_dict[model_name]

def init_selector(selector_name, x, y, manually_selected_features=None):
    n_features = None
    if selector_name == "SelectKBest":  # Order based on CoxPH c-index. Number of features optimized during tuning (based on C-index)
        selector_init = SelectKBest(k="all", score_func=fit_and_score_features)
        selector_init.fit(x, y)
        ordered_indices = np.argsort(selector_init.scores_)[::-1]  # Pre-compute features order
        selector = SelectKBestCustom(feature_ordered_indices_=ordered_indices)
    elif selector_name == "Manual":
        selector = ManualSelector(features=manually_selected_features)
    else:
        raise ValueError(f"Selector {selector_name} not valid")
    return selector, n_features
