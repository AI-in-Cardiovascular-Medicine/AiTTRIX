import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import shap
from survshap import SurvivalModelExplainer, PredictSurvSHAP
import matplotlib.pyplot as plt
import pickle

class SurvivalSHAPExplainer:
    """Wrapper to compute SurvSHAP values for survival models."""

    def __init__(
            self,
            model,
            x_train,
            y_train,
            eval_times,
            n_train_samples=100,
            calculation_method="shap_kernel",
            aggregation_method="integral",
            function_type="sf",
            n_jobs=10,
            random_state=42
    ):
        """
        Initialize the SurvSHAP explainer with a model and training data.

        Args:
            model (object): Fitted survival model to explain.
            x_train (pd.DataFrame): Training features.
            y_train (np.ndarray): Training labels (structured array with event/time).
            eval_times (array-like): Time points at which SurvSHAP values are computed.
            n_train_samples (int, optional): Number of training samples used for the explainer. Defaults to 100.
            calculation_method (str, optional): Method for SurvSHAP calculation. Defaults to "shap_kernel".
            aggregation_method (str, optional): Aggregation method for SurvSHAP values. Defaults to "integral".
            function_type (str, optional): Type of function to explain ("sf" for survival, "chf" for cumulative hazard). Defaults to "sf".
            n_jobs (int, optional): Number of parallel jobs. Defaults to 4.
            random_state (int, optional): Random seed. Defaults to 42.
        """
        self.model = model
        idx = shap.sample(x_train, n_train_samples, random_state=42).index
        self.x_train = x_train.loc[idx]
        self.y_train = y_train[idx]  # structured array
        self.eval_times = eval_times
        self.calc_method = calculation_method
        self.agg_method = aggregation_method
        self.function_type = function_type
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.explainer = SurvivalModelExplainer(  # Build the explainer
            model=self.model,
            data=self.x_train,
            y=self.y_train
        )
        self.features = np.sort(x_train.columns)

    def explain_single(self, obs, columns):
        """
        Compute SurvSHAP for a single observation.

        Args:
            obs (array-like): Feature values for a single observation.
            columns (Index or list): Column names of the features.

        Returns:
            tuple:
                shap_row (np.ndarray): SHAP values for this observation.
                feat_row (np.ndarray): Feature values for this observation.
                survshap (PredictSurvSHAP): SurvSHAP object for this observation.
        """
        xx = pd.DataFrame([obs], columns=columns)
        survshap = PredictSurvSHAP(
            calculation_method=self.calc_method,
            aggregation_method=self.agg_method,
            function_type=self.function_type,
            random_state=self.random_state
        )
        survshap.fit(
            explainer=self.explainer,
            new_observation=xx,
            timestamps=self.eval_times
        )
        sorted_res = survshap.result.sort_values("variable_name")
        shap_rows = {
            "aggregated": sorted_res["aggregated_change"].values,
            # "t = 365.0": sorted_res["t = 365.0"].values,
            # "t = 730.0": sorted_res["t = 730.0"].values,
            # "t = 1095.0": sorted_res["t = 1095.0"].values
        }
        feat_row = sorted_res["variable_value"].values
        return shap_rows, feat_row, survshap

    def explain_all(self, x_test):
        """
        Compute SurvSHAP for all observations in x_test and store results internally.

        Args:
            x_test (pd.DataFrame): Test features to explain.

        Returns:
            None: Results are stored in `self.shap_matrix`, `self.feature_matrix`, and `self.surv_objects`.
        """
        n_obs = len(x_test)
        columns = x_test.columns
        results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self.explain_single)(obs, columns)
            for obs in x_test.values
        )
        # results = []
        # for obs in x_test.values:
        #     results.append(self.explain_single(obs, columns))
        # Pre-allocate outputs
        shap_rows_example, _, _ = results[0]
        keys = shap_rows_example.keys()
        df_shaps = {k: pd.DataFrame(0.0, index=x_test.index, columns=self.features) for k in keys}
        df_feat = pd.DataFrame(0.0, index=x_test.index, columns=self.features)
        surv_objects = [None] * n_obs
        # Assign results
        for i, (shap_rows, feat_row, survshap) in enumerate(results):
            for k in keys:
                df_shaps[k].iloc[i] = shap_rows[k]
            df_feat.iloc[i] = feat_row
            surv_objects[i] = survshap
        # Store internally
        self.shap_matrices = df_shaps
        self.feature_matrix = df_feat
        self.surv_objects = surv_objects

    def plot_summary(self, feature_map=None, max_display=100):
        """
        Create a summary SHAP beeswarm plot of all explained observations.

        Args:
            max_display (int, optional): Maximum number of features to display. Defaults to 100.
            feature_map (dict, optional): Feature map to use. Defaults to None.

        Returns:
            matplotlib.figure.Figure: Figure object containing the SHAP summary plot.
        """
        if self.shap_matrices is None or self.feature_matrix is None:
            raise ValueError("You must call .explain() or .load_shap_values() before plotting.")
        if feature_map is not None:
            column_names = [feature_map[col] for col in self.features]
        else:
            column_names = self.features
        labels = ['aggregated']#, '1y', '2y', '3y']
        figs = []
        for k in self.shap_matrices.keys():
            shap_values = shap.Explanation(
                values=-self.shap_matrices[k].values,
                data=self.feature_matrix.values,
                feature_names=column_names,
                base_values=np.zeros(len(self.shap_matrices[k]))
            )
            fig, ax = plt.subplots(figsize=(9, 7))
            from matplotlib.colors import LinearSegmentedColormap
            base = plt.cm.plasma
            colors = base(np.linspace(0.0, 0.90, 256))  # cut top 20%
            plasma_muted = LinearSegmentedColormap.from_list(
                "plasma_muted", colors
            )
            shap.plots.beeswarm(
                shap_values,
                max_display=max_display,
                show=False,
                plot_size=None,
                color=plasma_muted,
            )
            plt.yticks(fontsize=18, fontweight="bold")
            plt.xlabel("SHAP value (impact on model output)", fontsize=13)
            plt.tight_layout()
            figs.append(fig)
        return figs, labels

    def save_shap_values(self, out_path):
        """
        Save SHAP values and feature values to a pickle file.

        Args:
            out_path (str): Path to save the pickled dictionary containing:
                - "shap_values": SHAP values dataframe
                - "feature_values": feature values dataframe
        """
        out_dict = {
            "shap_values": self.shap_matrices,
            "feature_values": self.feature_matrix
        }
        with open(out_path, "wb") as f:
            pickle.dump(out_dict, f)

    def load_shap_values(self, in_path):
        """
        Load precomputed SHAP and feature values from a pickle file.

        Args:
            in_path (str): Path to the pickle file.
        """
        with open(in_path, "rb") as f:
            data = pickle.load(f)
        self.shap_matrices = data["shap_values"]
        self.feature_matrix = data["feature_values"]
