import numpy as np
from loguru import logger
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from evaluation.calibration import mean_calibration, ici_survival_times
from evaluation.discrimination import antolini_concordance_index
from sklearn.utils import resample  # for Bootstrap sampling
from joblib import Parallel, delayed


class BaseEvaluator:
    @staticmethod
    def _raise_error(msg):
        logger.error(msg)
        raise ValueError(msg)

    def evaluate_model(self, risk, risk_time, y, suffix):
        """
        Evaluates a survival model on validation and test sets using multiple metrics. Bootstrapped confidence
        intervals are computed.

        Parameters:
        - estimator: Trained survival model.

        Returns:
        - Dictionary of evaluation metrics for validation and test sets.
        """
        np.random.seed(self.seed)
        # Compute metrics
        metrics_dict = {
            'evaluation_times': self.eval_times.tolist(),
            'truncation_time': self.tau,
            'bootstrap_iterations': self.bootstrap_iterations,
        }
        metrics = self.custom_survival_scorer(estimator=None, X=None, y=y, risk_scores=risk, risk_at_times=risk_time, suffix=suffix)
        metrics_dict.update(metrics)
        # logger.info(f"{' '*35}{suffix}: computed evaluation metrics, now bootstrap...")
        metrics_dict.update(self.bootstrap(risk, risk_time, y, suffix=suffix))  # bootstrap CI
        return metrics_dict

    def get_risk_at_eval_times(self, estimator, x):
        """
        Computes risk scores at predefined evaluation times.

        Parameters:
        - estimator: Trained survival model.
        - x: Input features.

        Returns:
        - Array of risk scores (1 - survival probability) at each evaluation time.
        """
        surv_func = estimator.predict_survival_function(x)
        risk_scores = np.array([[1 - func(t) for t in self.eval_times] for func in surv_func])  # risk = 1 - surv
        return risk_scores

    @staticmethod
    def bootstrap_step(risk, risk_time, y, y_train, eval_times, tau, time_column, event_column, calibration_method):
        """
        Computes bootstrap estimates of survival metrics on resampled data. Static method to allow parallel processing.

        Parameters:
        - risk: Array of risk scores.
        - risk_time: Array of risk scores at evaluation times.
        - y: Test set survival data (DataFrame).
        - y_train: Training set survival data (for IPCW-based metrics).
        - eval_times: Array of evaluation time points.
        - tau: Truncation time for Uno's c-index.
        - time_column: Column name for event times.
        - event_column: Column name for event indicators.

        Returns:
        - Tuple of computed metrics: Harrell's c-index, Uno's c-index, Antolini's c-index, time-dependent AUC,
          mean calibration, and ICI.
        """
        # Sampling with replacement
        idx_resample = resample(np.arange(len(y)))
        y_resampled = y[idx_resample]
        risk_resampled = risk[idx_resample]
        risk_time_resampled = risk_time[idx_resample]
        # Compute metrics
        cindex_harrell = concordance_index_censored(y_resampled[event_column], y_resampled[time_column],
                                                    risk_resampled)[0]
        cindex_uno = concordance_index_ipcw(y_train, y_resampled, risk_resampled, tau=tau)[0]
        # check if eval times are within max observation time
        eval_times_outside = eval_times > y_resampled[time_column].max()
        if any(eval_times_outside):
            eval_times = eval_times[eval_times <= y_resampled[time_column].max()]
            risk_time_resampled = risk_time_resampled[:, :len(eval_times)]
        cindex_antolini = antolini_concordance_index(durations=y_resampled[time_column],
                                                     labels=y_resampled[event_column], cuts=eval_times,
                                                     risk=risk_time_resampled, time_max=np.max(eval_times))
        roc_auc = cumulative_dynamic_auc(y_train, y_resampled, risk_time_resampled, eval_times)[0]
        mean_calib = mean_calibration(durations=y_resampled[time_column], eval_times=eval_times,
                                      events=y_resampled[event_column], risk_scores=risk_time_resampled).values
        ici = ici_survival_times(durations=y_resampled[time_column], events=y_resampled[event_column], parallel=False,
                                 risk_times=risk_time_resampled, times=eval_times, method=calibration_method)
        if any(eval_times_outside):
            n_nan_to_add = sum(eval_times_outside)
            roc_auc = np.insert(roc_auc, len(roc_auc), [np.nan] * n_nan_to_add)
            mean_calib = np.insert(mean_calib, len(mean_calib), [np.nan] * n_nan_to_add)
            ici = np.insert(ici, len(ici), [np.nan] * n_nan_to_add)
        # brier_score = integrated_brier_score(y_train, y_resampled, 1 - risk_time_resampled, eval_times)
        return cindex_harrell, cindex_uno, cindex_antolini, roc_auc, mean_calib, ici  # , brier_score

    def bootstrap(self, risk, risk_time, y, suffix: str = None):
        """
        Computes bootstrap confidence intervals for survival metrics.

        Parameters:
        - risk: Array of risk scores.
        - risk_time: Risk scores at evaluation times.
        - y: SurvivalExperiment labels for the evaluation set.
        - suffix: Optional suffix to append to result keys (e.g. "val" or "test").

        Returns:
        - Dictionary of median and 95% confidence intervals for c-indexes, AUC,
          mean calibration, and ICI.
        """
        args = (risk, risk_time, y, self.y_train, self.eval_times, self.tau, self.time_column, self.event_column, self.calibration_method)
        boot_results = Parallel(n_jobs=-1)(delayed(self.bootstrap_step)(*args) for _ in range(self.bootstrap_iterations))
        # boot_results = []
        # for _ in range(self.bootstrap_iterations):
        #     boot_results.append(self.bootstrap_step(*args))

        # Define metric names and their corresponding indices in boot_results
        metrics = {
            'harrell': 0,
            'uno': 1,
            'ant': 2,
            'auc': 3,
            'mCalib': 4,
            'ici': 5,
            # 'brier_score': 6
        }

        # Computing mean and CI
        ci_dict = {}
        for metric, idx in metrics.items():
            boot_values = np.array([result[idx] for result in boot_results])
            if np.isnan(boot_values).any():
                logger.warning(f"{' '*39}{np.isnan(boot_values).sum()} NaN values found in bootstrap results for {metric}. They will be ignored in CI computation.")
            ci_value = [
                np.nanquantile(boot_values, 0.025, axis=0),
                np.nanquantile(boot_values, 0.975, axis=0)
            ]
            median = np.quantile(boot_values, 0.5, axis=0)
            setattr(self, f"{metric}_mean", np.mean(boot_values, axis=0))
            setattr(self, f"{metric}_ci", ci_value)
            # Save to metrics dictionary
            if metric in ['auc', 'mCalib', 'ici']:
                for i, time_name in enumerate(self.eval_times_names):
                    ci_dict[f"CI_{metric}_{time_name}"] = [float(np.round(ci_value[0][i], 5)),
                                                           float(np.round(ci_value[1][i], 5))]
                    ci_dict[f"{metric}_{time_name}_median_boot"] = np.round(median[i], 5)
            else:
                ci_dict[f"CI_{metric}"] = [float(np.round(ci_value[0], decimals=5)),
                                                float(np.round(ci_value[1], decimals=5))]
                ci_dict[f"{metric}_median_boot"] = np.round(median, 5)
        if suffix:  # Add suffix to keys if required
            ci_dict = {f"{k}_{suffix}": v for k, v in ci_dict.items()}
        return ci_dict

    def custom_survival_scorer(self, estimator, X, y, risk_scores=None, risk_at_times=None, suffix: str = None):
        """
        Custom scoring function for survival models, computing:
        - Harrell's, UNo's and Antolini's C-index
        - Cumulative Dynamic AUC
        - Mean calibration
        - Integrated Calibration Index (ICI)

        Args:
            estimator: A fitted survival model following scikit-survival API.
            X: Feature matrix.
            y: SurvivalExperiment structured array or DataFrame (event indicator and time).
            risk_scores: Optional, pre-computed risk scores.
            risk_at_times: Optional, pre-computed time-dependent risk scores.
            suffix: Optional suffix to append to metric names.

        Returns:
            A dictionary containing survival evaluation metrics.
        """
        # --- Input check ---
        if estimator is None and risk_scores is None:
            msg = "Either an estimator must be provided to compute risk scores, or precomputed `risk_scores` must be supplied."
            self._raise_error(msg)
        if y is None:
            msg = "Argument `y` (survival outcomes) must be provided."
            self._raise_error(msg)
        # --- Compute risk scores ---
        if risk_scores is None:
            if X is None:
                msg = "`X` must be provided if estimator is used to compute risk scores."
                self._raise_error(msg)
            risk_scores = estimator.predict(X)
        # --- Determine risk_at_times ---
        if hasattr(estimator, "predict_survival_function") or risk_at_times is not None:
            # Model supports survival prediction OR precomputed risk curves are given
            if risk_at_times is None:
                surv_funcs = estimator.predict_survival_function(X)
                risk_at_times = np.array([[1 - sf(t) for t in self.eval_times] for sf in surv_funcs])
            time_dependent = True
        else:
            # No survival prediction support → use constant risk over all times
            risk_at_times = np.tile(risk_scores[:, None], (1, len(self.eval_times)))
            time_dependent = False
        # --- Compute metrics ---
        cindex_harrell = concordance_index_censored(y[self.event_column], y[self.time_column], risk_scores)[0]  # Harrell's C-index
        cindex_uno = concordance_index_ipcw(self.y_train, y, risk_scores, tau=self.tau)[0]  # Uno's C-index (IPCW-adjusted)
        auc_values, _ = cumulative_dynamic_auc(self.y_train, y, risk_at_times, self.eval_times)
        mean_calibration_values = mean_calibration(durations=y[self.time_column], events=y[self.event_column],
                                                   risk_scores=risk_at_times, eval_times=self.eval_times)
        ici_values = ici_survival_times(durations=y[self.time_column], events=y[self.event_column],
                                        risk_times=risk_at_times, times=self.eval_times, method=self.calibration_method)
        if time_dependent:
            cindex_antolini = antolini_concordance_index(durations=y[self.time_column], labels=y[self.event_column],
                                                         cuts=self.eval_times, risk=risk_at_times,
                                                         time_max=np.max(self.eval_times))
            brier_score = integrated_brier_score(self.y_train, y, 1-risk_at_times, self.eval_times)
        else:
            cindex_antolini, brier_score = None, None
        # --- Store results ---
        metrics_dict = {
            "harrell": cindex_harrell,
            "uno": cindex_uno,
            "ant": cindex_antolini,
            "brier_score": brier_score
        }
        for i, time_name in enumerate(self.eval_times_names):
            metrics_dict.update({
                f"auc_{time_name}": auc_values[i],
                f"mCalib_{time_name}": mean_calibration_values.values[i],
                f"ici_{time_name}": ici_values[i]
            })
        if suffix:  # Add suffix to keys if required
            metrics_dict = {f"{k}_{suffix}": v for k, v in metrics_dict.items()}
        return metrics_dict
