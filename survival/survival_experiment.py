import os
from os.path import join
import sys
import pickle
import time
import json

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from omegaconf import OmegaConf
from sklearn.preprocessing import RobustScaler

from survival.init_estimators import init_model, init_selector
from survival.hyperparameters_search import set_params_search_space, set_hyperparams_optimizer
from helpers.nested_dict import NestedDefaultDict
from survival.base import BaseEvaluator
from survival.utils import stratified_split


class SurvivalExperiment(BaseEvaluator):
    def __init__(self, config, progress_manager=None) -> None:
        super().__init__()
        self.progress_manager = progress_manager
        self.out_dir = config.meta.out_dir
        self.table_file = join(self.out_dir, 'results_table.xlsx')
        self.results_file = join(self.out_dir, 'results.pkl')
        self.suffixes_test = config.meta.suffixes_test
        self.event_column = config.meta.events
        self.time_column = config.meta.times
        self.sex_col = config.meta.sex_column
        self.n_seeds = config.meta.n_seeds
        self.n_workers = config.meta.n_workers
        self.scaler_name = "RobustScaler"
        self.selectors = [sel for sel in config.survival.feature_selectors if config.survival.feature_selectors[sel]]
        self.models = [model for model in config.survival.models if config.survival.models[model]]
        self.n_cv_splits = config.survival.n_cv_splits
        self.n_iter_search = config.survival.n_iter_search
        self.max_features_to_select = config.survival.max_features_to_select
        self.min_features_to_select = config.survival.min_features_to_select
        self.eval_times = np.array(config.evaluation.eval_times)
        self.eval_times_names = config.evaluation.eval_times_names
        self.tau = self.eval_times[-1]  # truncation time (used only for computing Uno's c-index)
        self.bootstrap_iterations = config.evaluation.bootstrap_iterations
        self._load_manual_features(config)
        self.total_combinations = (
            self.n_seeds
            * len(self.selectors)
            * len(self.models)
        )
        # Create table columns
        set_names = ["val", "test"] if self.suffixes_test is None else ["val"] + self.suffixes_test
        train_harrell_col = ["harrell_train"]
        base_cols = self._expand_metrics(metrics=["harrell", "uno", "ant"], sets=set_names, add_ci=True)
        time_cols = self._expand_metrics(metrics=["auc", "mCalib", "ici"], sets=set_names, times=self.eval_times_names, add_ci=True)
        fixed_cols = ["Scaler", "Selector", "Model", 'search', 'train_strategy', "harrell_cv_val", "CI_harrell_cv_val"]
        extra_cols = ['evaluation_times', 'truncation_time', 'n_iter_search', 'bootstrap_iterations', "Seed", "n_selected_features"]
        self.result_cols = fixed_cols + train_harrell_col +  base_cols + time_cols + extra_cols  # Combine everything
        self.results_table = pd.DataFrame(
            index=range(self.total_combinations),
            columns=self.result_cols,
        )
        self.row_to_write = 0
        self.results = NestedDefaultDict()

    def __call__(self, seed, x_train, y_train, x_tests, y_tests):
        # Initialize parameters
        self.seed = seed
        self.x_train = x_train
        self.y_train = y_train
        self.x_tests = x_tests if isinstance(x_tests, list) else [x_tests]
        self.y_tests = y_tests if isinstance(y_tests, list) else [y_tests]
        self.results[self.seed]['x_train'] = x_train
        self.results[self.seed]['y_train'] = y_train
        self.results[self.seed]['x_tests'] = x_tests
        self.results[self.seed]['y_tests'] = y_tests
        self.results[self.seed]['eval_times'] = self.eval_times
        self.results[self.seed]['eval_times_names'] = self.eval_times_names
        # Update manually selected features based on preprocessing
        if self.manually_selected_features is not None:
            for name, feature_list in self.manually_selected_features.items():
                self.manually_selected_features[name] = np.intersect1d(feature_list, x_train.columns.values).tolist()
        # Fit and evaluate pipeline
        self.fit_and_evaluate_pipeline()

        return self.results_table

    def fit_selector(self, selector_name, scaler, estimator, stratified_folds):
        """
        Fits the feature selector and returns the fitted selector and the number of selected features.
        If an error occurs during fitting, it logs the error and returns None, None.
        """
        selector_name_complete = selector_name
        if "Manual" in selector_name:
            selector_name = "Manual"
            manual_features_set = self.manually_selected_features[selector_name_complete]
        else:
            manual_features_set = None
        indent = " " * 17
        if selector_name == "SelectKBest":
            logger.info(f"{indent}Fitting {selector_name} feature selector")
        tic = time.time()
        selector, n_selected_features = init_selector(
            x=scaler.fit_transform(self.x_train), y=self.y_train, manually_selected_features=manual_features_set,
            selector_name=selector_name)
        selector.set_output(transform="pandas")
        if selector_name == "SelectKBest":
            logger.info(f"{indent}Fitting-time {time.time() - tic}")
            ordered_features = list(self.x_train.columns[selector.feature_ordered_indices_])
            out_path = os.path.join(self.out_dir, f"{selector_name}_order.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(ordered_features, f, indent=2)
        return selector

    def fit_and_evaluate_pipeline(self):
        pbar = self.progress_manager.counter(
            total=self.total_combinations, desc="Training and evaluating all combinations", unit='it', leave=False
        )
        search = "rand"
        strategy = "refit"
        # Truncate y_test if necessary (AUC requires test times to lie within the range of train times)
        max_t = self.y_train[self.time_column].max()
        for i, y_test in enumerate(self.y_tests):
            mask = y_test[self.time_column] >= max_t
            if sum(y_test[self.event_column][mask]) > 0:
                logger.info(f"Truncated {sum(y_test[self.event_column][mask])} events to {max_t} days for AUC calculation")
            self.y_tests[i][self.time_column][mask] = max_t
            self.y_tests[i][self.event_column][mask] = 0  # Set events to 0 (event after max censoring)
        # Create CV splits
        self.folds = stratified_split(
            y=self.y_train,
            time_col=self.time_column,
            event_col=self.event_column,
            n_splits=self.n_cv_splits,
            seed=self.seed)
        self._check_cv_folds()
        self.results[self.seed]["folds_indices"] = self.folds
        best_params_all = []
        # Loop over models
        for model_name in self.models:
            estimator = init_model(self.seed, self.n_workers, model_name)
            logger.info(f"\033[95m{model_name} model\033[0m")
            scaler = RobustScaler()
            scaler = scaler.set_output(transform="pandas")
            for selector_name in self.selectors:  # Loop over Selectors
                logger.info(f"    \033[92m{selector_name} selector\033[0m")
                selector_name_complete = selector_name
                selector = self.fit_selector(selector_name_complete, scaler, estimator, self.folds)
                selector_name_complete = selector_name
                if "Manual" in selector_name:
                    selector_name = "Manual"
                config_log_msg = f"{selector_name_complete} - {model_name}"
                tic = time.time()
                logger.info(f"      Training {model_name} - {selector_name_complete} - {search}")
                # Create pipeline and parameter grid
                pipe = Pipeline(
                    [
                        ('scaler', scaler),
                        ("selector", selector),
                        ("model", estimator),
                    ]
                )
                # Hyperparameters search and fitting
                self.model_params, self.selector_params = set_params_search_space(
                    max_n_features=self.max_features_to_select,
                    min_features=self.min_features_to_select
                )
                param_grid = {**self.selector_params[selector_name], **self.model_params[model_name]}
                gcv = set_hyperparams_optimizer(
                    pipeline=pipe,
                    param_grid=param_grid,
                    n_iter_search=self.n_iter_search,
                    stratified_folds=self.folds,
                    n_workers=self.n_workers,
                    seed=self.seed,
                    eval_times=self.eval_times
                )
                gcv.fit(self.x_train, self.y_train)
                best_params = {k.replace("estimator__", ""): v for k, v in gcv.best_params_.items()}
                pipe.set_params(**best_params)
                # Get scores on validation sets and save
                validation_scores = cross_validate(pipe, self.x_train, self.y_train, cv=self.folds,
                                                   scoring=self.custom_survival_scorer, n_jobs=1,
                                                   return_estimator=True)
                refitted_model = gcv.best_estimator_  # extract best estimator
                refitted_model = refitted_model.estimator  # extract pipeline from wrapper
                logger.info(f'      Best params: {dict(best_params)}')
                best_params.update({"Selector": selector_name_complete, "Model": model_name})
                best_params_all.append(best_params)  # Store best parameters
                # Save validation risk
                risk_all = self.get_validation_risk(models=validation_scores["estimator"],
                                                    cv_splits=self.folds, scaler=scaler,
                                                    selector=selector)
                self.results[self.seed]["val_risk"][self.scaler_name][selector_name_complete][model_name][
                    search] = risk_all
                # Evaluate refitted model
                logger.info(f'      Evaluating {config_log_msg}')
                row = {"Seed": self.seed, "Selector": selector_name_complete, "Model": model_name,
                       "n_iter_search": self.n_iter_search,
                       "n_selected_features": len(refitted_model["model"].feature_names_in_)}
                metrics_cv = self.process_val_dict(validation_scores)
                row.update(metrics_cv)
                self.results[self.seed][self.scaler_name][selector_name_complete][model_name][search][strategy] = refitted_model
                metrics = self.evaluate_model(self.risk_val, self.risk_time_val, self.y_val, suffix="val")
                # Train metrics
                risk_train = refitted_model.predict(self.x_train)
                risk_times_train = self.get_risk_at_eval_times(refitted_model, self.x_train)
                metrics_train = self.evaluate_model(risk_train, risk_times_train, self.y_train, suffix="train")
                metrics.update(metrics_train)
                # Test metrics
                for x_test, y_test, suffix in zip(self.x_tests, self.y_tests, self.suffixes_test):
                    risk_test_ = refitted_model.predict(x_test)  # Compute risk scores on test
                    risk_time_test_ = self.get_risk_at_eval_times(refitted_model, x_test)
                    metrics_test_ = self.evaluate_model(risk_test_, risk_time_test_, y_test, suffix)
                    metrics.update(metrics_test_)
                row.update(metrics)
                self.results_table.loc[self.row_to_write] = row
                for column in self.results_table.columns:
                    self.results_table[column] = pd.to_numeric(self.results_table[column], errors='ignore')
                self.results_table[self.results_table.select_dtypes(
                    include='number').columns] = self.results_table.select_dtypes(include='number').round(5)
                self.row_to_write += 1
                self.results_table = self.results_table.sort_values(["Seed", "Selector", "Model", "train_strategy"])
                logger.info(f'      Saving results to {self.out_dir}')
                try:  # ensure that intermediate results are not corrupted by KeyboardInterrupt
                    self.save_results()
                except KeyboardInterrupt:
                    logger.warning('Keyboard interrupt detected, saving results before exiting...')
                    self.save_results()
                    sys.exit(130)
                pbar.update()
                logger.info(f"      Elapsed time {time.time() - tic}")
        best_params_df = pd.DataFrame(best_params_all)
        best_params_df.to_excel(join(self.out_dir, 'selected_params.xlsx'), index=False)
        pbar.close()

    def _check_cv_folds(self):
        logger.info("Stratified CV folds created:")
        # Print fold sizes and median time to event and to censoring
        for i, fold in enumerate(self.folds):
            y_fold = self.y_train[fold[1]]
            n_events = y_fold[self.event_column].sum()
            n_censored = len(y_fold) - n_events
            median_event_time = np.median(y_fold[self.time_column][y_fold[self.event_column] == 1])
            median_censor_time = np.median(y_fold[self.time_column][y_fold[self.event_column] == 0])
            logger.info(f"Fold {i + 1}: {len(fold[1])} samples, {n_events} events, {n_censored} censored, "
                        f"median event time: {median_event_time}, median censor time: {median_censor_time}")

    @staticmethod
    def process_val_dict(validation_scores):
        metrics_cv = {}
        for key, value in validation_scores.items():
            if key.startswith('test_'):
                mean = float(np.mean(value))
                std = np.std(value)
                ci_lower = mean - std * 1.96 / np.sqrt(len(value))
                ci_upper = mean + std * 1.96 / np.sqrt(len(value))
                mean_key = key[5:] + '_cv_val'  # Remove 'test_' and add '_val_mean'
                ci_key = "CI_" + key[5:] + '_cv_val'  # Remove 'test_' and add '_val_ci'
                metrics_cv[mean_key] = mean
                metrics_cv[ci_key] = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
        return metrics_cv

    def get_validation_risk(self, models, cv_splits, scaler, selector):
        x_train_preprocessed = scaler.transform(self.x_train)
        x_train_preprocessed = selector.transform(x_train_preprocessed)
        risk_times_all = []
        risk_all = []
        y_all = []
        sex_all = []
        for i, model in enumerate([m["model"] for m in models]):
            x_val = x_train_preprocessed.iloc[cv_splits[i][1]]
            y_val = self.y_train[cv_splits[i][1]]
            risk_times_val = self.get_risk_at_eval_times(model, x_val)
            risk_times_all.append(risk_times_val)
            risk_all.append(model.predict(x_val))
            y_all.append(y_val)
            if self.sex_col is not None:
                sex = self.x_train.iloc[cv_splits[i][1]][self.sex_col].values
                sex_all.append(sex)
        self.risk_time_val = np.concatenate(risk_times_all)
        self.risk_val = np.concatenate(risk_all)
        self.y_val = np.concatenate(y_all)
        risk_times_all = pd.DataFrame(self.risk_time_val, columns=self.eval_times_names)
        y_all = pd.concat([pd.DataFrame(a) for a in y_all], ignore_index=True)
        risk_times_all = pd.concat([risk_times_all, y_all], axis=1)
        risk_times_all["risk"] = self.risk_val
        if self.sex_col is not None:
            risk_times_all["sex"] = np.concatenate(sex_all)
        return risk_times_all

    @staticmethod
    def _expand_metrics(metrics, sets, times=None, add_ci=False):
        """
        Helper to create column names: expand metrics to include sets, optionally time points and CI variants.
        """
        if add_ci:
            metrics = metrics + [f"CI_{m}" for m in metrics]
        if times is None:
            return [f"{m}_{s}" for m in metrics for s in sets]
        else:
            return [f"{m}_{t}_{s}" for m in metrics for s in sets for t in times]

    def _load_manual_features(self, config):
        """
        Load manually selected features from paths specified in the config.
        Updates self.manually_selected_features and self.selectors accordingly.
        """
        if "Manual" not in self.selectors:
            self.manually_selected_features = None
            return
        # Remove placeholder "Manual" from selectors
        self.selectors.remove("Manual")
        paths_manual_features = config.meta.paths_manually_selected_features
        if paths_manual_features is None:
            logger.info(
                "'Manual' feature selection needs config.meta.paths_manually_selected_features to be specified. "
                "Manual feature selection will not be performed."
            )
            self.manually_selected_features = None
            return
        elif not OmegaConf.is_list(paths_manual_features):
            logger.info(
                "config.meta.paths_manually_selected_features needs to be a list of paths if 'Manual' is specified. "
                "Manual feature selection will not be performed."
            )
            self.manually_selected_features = None
            return

        names_manual_features = config.meta.names_manually_selected_features
        if OmegaConf.is_list(names_manual_features) and len(names_manual_features) == len(paths_manual_features):
            names_manual_features = [f"Manual_{name}" for name in names_manual_features]
        else:
            logger.info(
                "config.meta.names_manually_selected_features is not specified or has incorrect length. "
                "Set selection names to Manual_1, Manual_2, ..."
            )
            names_manual_features = [f"Manual_{i + 1}" for i in range(len(paths_manual_features))]

        self.manually_selected_features = {}
        for path, name in zip(paths_manual_features, names_manual_features):
            with open(path, 'r') as f:
                self.manually_selected_features[name] = json.load(f)
            self.selectors.append(name)


    def save_results(self):
        os.makedirs(os.path.dirname(self.table_file), exist_ok=True)
        self.results_table.to_excel(self.table_file, index=False)
        with open(self.results_file, 'wb') as file:
            pickle.dump(self.results, file)
