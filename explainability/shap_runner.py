import os
import pickle
import json
import re
import numpy as np
import pandas as pd
from explainability.shap_explainer import SurvivalSHAPExplainer
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from loguru import logger

class RunExplainability:
    """Class to compute SHAP explanations for survival models from a results folder."""

    def __init__(self, folder=None, config=None, combinations=None):
        """
        Initialize the explainability runner.

        Args:
            folder (str, optional): Path to results folder containing 'results.pkl'.
            config (OmegaConf object, optional): Hydra config object.
            combinations (list of dict, optional): List of model combinations to explain. If None, all from results table are used.
        """
        if folder is not None:
            self.folder = folder
            config_path = os.path.join(folder, "config.yaml")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Cannot find Hydra config at: {config_path}")
            self.config = OmegaConf.load(config_path)  # Load Hydra config
            print(f"Hydra-style config loaded from: {config_path}")
        elif config is not None:
            self.folder = config.meta.out_dir
            self.config = config
        else:
            raise ValueError("You must provide either `folder` or `config`.")

        self.suffixes_test = self.config.meta.suffixes_test
        self.results_file = os.path.join(self.folder, 'results.pkl')
        self.metrics_table = pd.read_excel(os.path.join(self.folder, 'results_table.xlsx'))
        self.seed = self.config.meta.seed
        self.shap_background_size = self.config.explainability.shap_background_size
        self.n_timestamps = self.config.explainability.n_timestamps
        self.combine = self.config.explainability.combine_test_sets
        self.n_jobs = 10
        feature_map_path = self.config.explainability.feature_map_path
        if feature_map_path is not None:
            with open(feature_map_path, 'rb') as f:
                self.feature_map = json.load(f)
            self.feature_map = {
                k: re.sub(r"\s*\([^)]*\)", "", v).strip()
                for k, v in self.feature_map.items()
            }
        else:
            self.feature_map = None

        # Load results and combinations
        results_file = os.path.join(self.folder, "results.pkl")
        with open(results_file, "rb") as f:
            self.results = pickle.load(f)

        eval_times = self.config.evaluation.eval_times
        self.eval_times = np.linspace(eval_times[0], eval_times[-1], self.n_timestamps)

        if combinations is None:
            # Default: use all combinations from metrics table
            self.combinations = list(
                self.metrics_table[["Selector", "Model"]]
                .itertuples(index=False, name=None)
            )
        else:
            # If it's a list of dicts, convert to tuples
            if isinstance(combinations, list) and combinations and isinstance(combinations[0], dict):
                self.combinations = [(d["selector"], d["model"]) for d in combinations]
            else:
                self.combinations = combinations  # assume already tuples or valid format

        # Extract x_train / y_train
        seed = self.config.meta.seed
        self.x_train = self.results[seed]["x_train"]
        self.y_train = self.results[seed]["y_train"]
        self.x_tests = self.results[seed]["x_tests"]
        self.y_tests = self.results[seed]["y_tests"]
        self.seed = seed
        self.scaler_name = "RobustScaler"
        self.search = "rand"
        self.train_strategy = "refit"

    def run(self):
        """
        Run SHAP explanations for specified combinations.
        """
        out_folder = os.path.join(self.folder, "explanations")
        os.makedirs(out_folder, exist_ok=True)

        if len(self.x_tests) == 1 and self.combine:
            logger.info("Only one test set. Do not combine SHAP values.")
            self.combine = False

        logger.info("Compute SHAP values:")
        for comb in self.combinations:
            selector, model_name = comb
            comb_log_msg = f"{selector} - {model_name}"
            try:
                logger.info(f"{' '*2}" + comb_log_msg)
                out_folder_comb = os.path.join(out_folder, "_".join(comb))
                os.makedirs(out_folder_comb, exist_ok=True)
                best_estimator = self.results[self.seed][self.scaler_name][selector][model_name][self.search][self.train_strategy]
                selector = best_estimator["selector"]
                scaler = best_estimator["scaler"]
                x_train = selector.transform(scaler.transform(self.x_train))
                model = best_estimator["model"]
                df_shap_list, df_feature_list = [], []
                for x_test, y_test, suffix in zip(self.x_tests, self.y_tests, self.suffixes_test):
                    # Create explainer
                    survshap_explainer = SurvivalSHAPExplainer(
                        model=model,
                        x_train=x_train,
                        y_train=self.y_train,
                        n_train_samples=self.shap_background_size,
                        eval_times=self.eval_times,
                        n_jobs=self.n_jobs,
                        random_state=self.seed
                    )
                    # Check if SHAP values were already computed
                    shap_path = os.path.join(out_folder_comb, f"shap_values_{suffix}.pkl")
                    plot_path = os.path.join(out_folder_comb, f"beeswarm_summary_{suffix}.png")
                    shap_exists = os.path.exists(shap_path)
                    plot_exists = os.path.exists(plot_path)
                    if shap_exists and plot_exists:
                        logger.info(f"{' '*4}[SKIP] SHAP values and plot already exist for {suffix}")
                        if self.combine:
                            survshap_explainer.load_shap_values(shap_path)  # Load SHAP values if already computed
                            df_shap_list.append(survshap_explainer.shap_matrices["aggregated"])
                            df_feature_list.append(survshap_explainer.feature_matrix)
                        continue
                    # Compute SHAP values
                    if shap_exists and not plot_exists:
                        logger.info(f"{' '*4}[LOAD + PLOT] Loading SHAP and creating plot for {suffix}")
                        survshap_explainer.load_shap_values(shap_path)  # Load SHAP values if already computed
                    else:
                        x_test_proc = selector.transform(scaler.transform(x_test))  # Compute SHAP values
                        survshap_explainer.explain_all(x_test_proc)
                    # Plot summary
                    figs, labels = survshap_explainer.plot_summary(feature_map=self.feature_map, max_display=100)
                    for fig, label in zip(figs, labels):
                        fig.savefig(os.path.join(out_folder_comb, f"beeswarm_summary_{label}_{suffix}.png"), dpi=300)
                        plt.close(fig)
                    # Save SHAP values
                    out_path = os.path.join(out_folder_comb, f"shap_values_{suffix}.pkl")
                    survshap_explainer.save_shap_values(out_path)
                    if self.combine:
                        df_shap_list.append(survshap_explainer.shap_matrices["aggregated"])
                        df_feature_list.append(survshap_explainer.feature_matrix)

                if self.combine:
                    df_shap_combined = pd.concat(df_shap_list)
                    df_feature_combined = pd.concat(df_feature_list)
                    # Create new explainer for combined data
                    survshap_explainer = SurvivalSHAPExplainer(
                        model=model,
                        x_train=x_train,
                        y_train=self.y_train,
                        eval_times=self.eval_times,
                        n_jobs=self.n_jobs,
                        random_state=self.seed
                    )
                    survshap_explainer.shap_matrices = {"aggregated": df_shap_combined}
                    survshap_explainer.feature_matrix = df_feature_combined
                    figs, labels = survshap_explainer.plot_summary(feature_map=self.feature_map)
                    for fig, label in zip(figs, labels):
                        fig.savefig(os.path.join(out_folder_comb, f"beeswarm_summary_{label}_all_tests_combined.png"))
                        plt.close(fig)
                    out_path_combined = os.path.join(out_folder_comb, f"shap_values_all_tests_combined.pkl")
                    survshap_explainer.save_shap_values(out_path_combined)
            except Exception as e:
                logger.error(f"Error encountered for {comb_log_msg}. Error message: {e}")
