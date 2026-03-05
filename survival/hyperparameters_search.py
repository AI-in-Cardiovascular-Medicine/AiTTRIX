from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sksurv.metrics import as_integrated_brier_score_scorer


def set_params_search_space(max_n_features=2, min_features=1):
    """
    Defines hyperparameter search space for survival models based on search_strategy.

    Parameters:
    - max_n_features: int, max number of features to select

    Returns:
    - model_params: dict, hyperparameter distributions for models
    - selector_params: dict, hyperparameter distributions for feature selectors
    """
    key = "estimator__"
    model_params = {
        "CoxPH": {
            f"{key}model__alpha": uniform(0.00001, 9.99999)
        },
        "RSF": {
            f"{key}model__n_estimators": randint(100, 200),
            f"{key}model__max_depth": randint(3, 10),
            f"{key}model__min_samples_split": randint(5, 10),
            f"{key}model__min_samples_leaf": randint(3, 10),
            f"{key}model__max_features": ["log2", "sqrt", None],
            f"{key}model__max_samples": uniform(0.4, 0.3)
        },
    }
    selector_params = {
        "SelectKBest": {
            f"{key}selector__k": randint(min_features, max_n_features)
        },
        "Manual": {}
    }
    return model_params, selector_params


def set_hyperparams_optimizer(pipeline, param_grid, n_iter_search, stratified_folds, n_workers, seed, eval_times=None):
    # Wrap pipeline to use Brier score as scorer
    pipeline_with_scorer = as_integrated_brier_score_scorer(pipeline, times=eval_times)
    optimizer = RandomizedSearchCV(
        estimator=pipeline_with_scorer,
        n_iter=n_iter_search,
        param_distributions=param_grid,
        n_jobs=n_workers,
        cv=stratified_folds,
        verbose=0,
        random_state=seed,
        error_score='raise'
    )
    return optimizer
