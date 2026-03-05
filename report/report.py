import os
import warnings
import json
import pickle
from loguru import logger
from ast import literal_eval

from itertools import combinations
from lifelines.statistics import logrank_test

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import seaborn as sns
from lifelines import KaplanMeierFitter, AalenJohansenFitter
from lifelines.plotting import add_at_risk_counts
from evaluation.calibration import calibration_plot_survival
from matplotlib.collections import LineCollection
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from dcurves import dca, plot_graphs

from preprocessing.cleaning_utils import get_continuous_categorical_features
import math

plt.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

class Report:
    def __init__(self, config) -> None:
        self.plot_format = config.meta.plot_format
        self.time_column = config.meta.times
        self.event_column = config.meta.events
        # self.set_names = ["val", "test"] if self.suffixes_test is None else ["val"] + self.suffixes_test
        suffixes_test = config.meta.suffixes_test
        self.set_names = ["swiss", "vienna"]
        self.ids_suffixes_to_keep = [i for i in range(len(suffixes_test)) if suffixes_test[i] in self.set_names]
        self.suffixes_test = [suffixes_test[i] for i in self.ids_suffixes_to_keep]
        self.experiment_dir = os.path.splitext(config.meta.train_file)[0] if config.meta.out_dir is None else config.meta.out_dir
        self.dpi = config.meta.plot_dpi
        self.calibration_curve_method = config.evaluation.calibration_curve_method
        self.is_competing = 1 if config.meta.competing_events is not None else 0
        self.results_file = os.path.join(self.experiment_dir, 'results.pkl')
        self.metrics_table = pd.read_excel(os.path.join(self.experiment_dir, 'results_table.xlsx'))
        self.seed = config.meta.seed
        np.random.seed(self.seed)
        if os.path.exists(os.path.join(self.experiment_dir, "best_combinations.json")):
            logger.info("Making plots only for best combinations.")
            with open(os.path.join(self.experiment_dir, "best_combinations.json"), "r") as file:
                self.best_combinations = json.load(file)
        else:
            self.best_combinations = None
        self.search_strategies = [strategy for strategy in config.survival.hyperparams_search_strategy
                                  if config.survival.hyperparams_search_strategy[strategy]]
        # self.colors = cm.tab10.colors
        self.colors = ["#41afaa", "#00a0e1", "#466eb4", "#e6a532", "#d7642c", "#af4b91"]
        self.colors = ["#41afaa", "#466eb4", "#e6a532", "#d7642c", "#af4b91"]
        self.green = "#2E8B57"  # tab10_colors[2] # Green
        self.red = "#C0504D"    # tab10_colors[3] # Red
        self.orange = "#FF8C42"
        self.colors = ["#9A3D3F", '#7FB77E']
        self.sets_label_dict = {
            "val": "Validation",
            "test_combined": "Test",
            "test_treated_first_fup": "Test (treated)",
            "vienna": "Vienna Test",
            "swiss": "Swiss Test",
            "swiss_vienna_treated": "Treated",
        }

    def __call__(self):
        with open(self.results_file, 'rb') as f:
            self.results = pickle.load(f)
        results = self.results[self.seed]
        self.out_dir = os.path.join(self.experiment_dir, "plots")
        os.makedirs(self.out_dir, exist_ok=True)
        self.x_train = results['x_train']
        self.y_train = results['y_train']
        self.x_tests = self.x_tests = [results['x_tests'][i] for i in self.ids_suffixes_to_keep]
        self.y_tests = [results['y_tests'][i] for i in self.ids_suffixes_to_keep]
        self.eval_times = results['eval_times']
        self.eval_times_names = results['eval_times_names']
        self.competing_events_label_test = results["comp_event_label_test"] if self.is_competing else [y_test[self.event_column] for y_test in self.y_tests]
        if self.best_combinations is not None:
            self.combs = [tuple(comb.values()) for comb in self.best_combinations]
        else:
            self.combs = list(self.metrics_table[["Scaler", "Selector", "Model", "search", "train_strategy"]].itertuples(index=False, name=None))
        self.auc_all = []
        self.auc_err_all = []
        self.calib_curves = []
        self.labels = []
        val_ids = np.concat([c[1] for c in results["folds_indices"]])
        self.x_val = self.x_train.iloc[val_ids]
        for comb in self.combs:
            try:
                scaler, selector, model, search, train_strategy = comb
                if results[scaler][selector][model][search][train_strategy] == {}:
                    logger.info(f"No trained model for Scaler={scaler}, Selector={selector},"
                                f"Model={model}, Search={search}.")
                    continue
                self.best_estimator = results[scaler][selector][model][search][train_strategy]
                # Get risk on the TRAINING set
                self.risk_times_train = self.get_risk_at_eval_times(self.best_estimator, self.x_train)
                self.risk_train = self.best_estimator.predict(self.x_train)
                # Get Risk scores
                self.risk_scores_test = []
                for i, test_name in enumerate(self.suffixes_test):
                    x_test_i = self.x_tests[i]
                    self.risk_scores_test.append(self.best_estimator.predict(x_test_i))
                df_val_risk = results["val_risk"][scaler][selector][model][search]
                self.risk_scores_val = results["val_risk"][scaler][selector][model][search]["risk"]
                self.y_val = self.to_structured_array(df_val_risk[[self.event_column, self.time_column]])
                if hasattr(self.best_estimator["model"], "predict_survival_function"):
                    self.risk_times_tests = []
                    for i, x_test_i in enumerate(self.x_tests):
                        surv_func = self.best_estimator.predict_survival_function(x_test_i)
                        risk_time = np.array([[1 - func(t) for t in self.eval_times] for func in surv_func])  # risk = 1 - surv
                        self.risk_times_tests.append(risk_time)
                    self.risk_times_val = df_val_risk[self.eval_times_names].values
                calib_curve = self.calibration_plot()
                self.calib_curves.append(calib_curve)
                self.labels.append(f"{selector}")
                self.decision_curve(comb)
            except Exception as e:
                logger.error(f"Error while processing combination {comb}: {e}")
                continue

        # Plot all calibration curves on the same plot
        self.calibration_plots_by_model()


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

    def km_cif_by_risk(self, scaler, selector, model, strategy, curve_type, n_groups=2, ci_show=False):
        logger.info(f"KM stratified by risk for Scaler={scaler}, Selector={selector},"
                                f"Model={model}, Search={strategy}")
        tab10_colors = cm.tab10.colors
        green = "#29AE61"  # tab10_colors[2] # Green in tab10
        red = "#E65047"    # tab10_colors[3] # Red in tab10
        n = n_groups - 2
        middle_colors = [c for i, c in enumerate(tab10_colors) if i not in [2, 3]][1:2]
        colors = [green] + middle_colors + [red] if n_groups > 2 else [green, red]
        n_sets = len(self.set_names)
        fig_multi, axes_multi = plt.subplots(1, n_sets, figsize=(6 * n_sets, 6), sharey=True)
        idx = 0
        grid = np.quantile(self.risk_times_train[:, 2], np.linspace(start=0, stop=1, num=n_groups + 1))
        for set_, risk_scores, y, comp_ev_label in zip(self.set_names,
                                                       [self.risk_times_val] + self.risk_times_tests,
                                                       [self.y_val] + self.y_tests,
                                                       [self.competing_events_label_val] + self.competing_events_label_test):
            # grid = np.quantile(risk_scores, np.linspace(start=0, stop=1, num=n_groups + 1))
            bins = np.digitize(risk_scores[:, 2], grid[1:-1], right=True)
            subsets = [(bins == i) for i in range(n_groups)]  # Create boolean masks for each group
            labels = ["Low Risk", "Medium Risk", "High Risk"] if n_groups == 3 else ["Low Risk", "High Risk"]
            curves = []
            curves_multi = []
            fig, ax = plt.subplots(figsize=(7, 6))
            ax_multi = axes_multi[idx]
            for subset, color, label in zip(subsets, colors, labels):
                curve = KaplanMeierFitter()
                curve.fit(durations=y[self.time_column][subset] / 365,
                          event_observed=y[self.event_column][subset],
                          label=label)
                curve.plot_survival_function(color=color, loc=slice(0, (self.eval_times[-1] + 10) / 365),
                                             ci_show=ci_show, linewidth=3)
                curve.plot_survival_function(ax=ax_multi, color=color, loc=slice(0, (self.eval_times[-1] + 10) / 365),
                                             ci_show=ci_show,linewidth=3)
                curves.append(curve)
                curves_multi.append(curve)

            filename_base = "km_by_risk" if curve_type == "km" else "cif_by_risk"
            ax.legend(loc="lower left")  # Add legend
            y_label = 'Survival Probability' if curve_type == "km" else "Cumulative Incidence"
            ax.set_xlabel('Years', fontsize=15)
            ax.set_ylabel(y_label, fontsize=15)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.yaxis.grid(True, color='gray', alpha=0.1, linestyle='-') # Horizontal gridlines
            ax.xaxis.grid(False)
            ax.spines['top'].set_visible(False) # Remove top and right spines
            ax.spines['right'].set_visible(False)
            ax.set_title(self.sets_label_dict[set_], fontsize=15, fontweight='bold')
            add_at_risk_counts(*curves, ax=ax, rows_to_show=["At risk"])
            plt.tight_layout()
            filename = f"{filename_base}_{scaler}_{selector}_{model}_{strategy}_{set_}.{self.plot_format}"
            plt.savefig(os.path.join(self.out_dir, filename), dpi=self.dpi)
            plt.close()

            ax_multi.set_title(self.sets_label_dict[set_], fontsize=15, fontweight='bold')
            ax_multi.set_xlabel('Years', fontsize=15)
            if idx == 0:
                ax_multi.set_ylabel(y_label, fontsize=15)
            ax_multi.tick_params(axis='x', labelsize=14)
            ax_multi.tick_params(axis='y', labelsize=14)
            ax_multi.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax_multi.yaxis.grid(True, color='gray', alpha=0.1, linestyle='-')
            ax_multi.xaxis.grid(False)
            ax_multi.spines['top'].set_visible(False)
            ax_multi.spines['right'].set_visible(False)
            add_at_risk_counts(*curves_multi, ax=ax_multi, rows_to_show=["At risk"])
            leg = ax_multi.get_legend()
            if leg is not None:
                leg.remove()

            logger.info(f"\t{set_}")
            groups = [0, 1, 2]
            for g1, g2 in combinations(groups, 2):
                result = logrank_test(
                    y[self.time_column][subsets[g1]],
                    y[self.time_column][subsets[g2]],
                    event_observed_A=y[self.event_column][subsets[g1]],
                    event_observed_B=y[self.event_column][subsets[g2]]
                )
                logger.info(f"\t\t{labels[g1]} vs {labels[g2]}: p-value = {result.p_value:.4f}, test statistic = {result.test_statistic:.2f}")

            idx += 1

        handles, labels_ = axes_multi[0].get_legend_handles_labels()
        fig_multi.legend(handles, labels_, loc="lower center", ncol=len(labels_))
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        filename_multi = f"{filename_base}_{scaler}_{selector}_{model}_{strategy}_ALLSETS.{self.plot_format}"
        fig_multi.savefig(os.path.join(self.out_dir, filename_multi), dpi=self.dpi)
        plt.close(fig_multi)

    def calibration_plot(self):
        calib_curves = {set_name: {} for set_name in self.set_names}
        for set_, risk_scores, y in zip(self.set_names,
                                        [self.risk_times_val] + self.risk_times_tests,
                                        [self.y_val] + self.y_tests):
            for i, time_name in enumerate(self.eval_times_names):
                fig, grid, predict_grid, risk = calibration_plot_survival(durations=y[self.time_column],
                                                                          events=y[self.event_column],
                                                                          risk=risk_scores[:, i],
                                                                          time=self.eval_times[i],
                                                                          method=self.calibration_curve_method)
                calib_curves[set_][time_name] = {
                    "grid": grid,
                    "predict_grid": predict_grid,
                    "risk": risk
                }
        return calib_curves

    def _plot_single_calibration_curve(
            self, ax1, ax2, calib_curve_t, label, color, lw_based_on_density=True):
        """Plot a single calibration curve + density plot onto given axes."""
        if lw_based_on_density:
            # Smooth spline
            x_orig = calib_curve_t["grid"]
            y_orig = calib_curve_t["predict_grid"]
            risk = calib_curve_t["risk"]
            x = np.linspace(x_orig.min(), x_orig.max(), 500)
            interp_fn = interp1d(x_orig, y_orig, kind="cubic")
            y = interp_fn(x)
            # KDE for line width
            kde = gaussian_kde(risk, bw_method=0.3)
            density = kde.evaluate(x)
            lw = 0.9 + 2.4 * (density - density.min()) / (density.max() - density.min())
            # Segments
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, linewidths=lw[:-1], color=color, label=label)
            ax1.add_collection(lc)
        else:
            ax1.plot(calib_curve_t["grid"], calib_curve_t["predict_grid"],
                     "-", linewidth=2, color=color, label=label)
        # Density (bottom subplot)
        sns.kdeplot(calib_curve_t["risk"], ax=ax2, color=color, bw_adjust=0.5, fill=True, alpha=0.15)

    def calibration_plots_by_model(self, lw_based_on_density=True):
        """
        For each model in self.calib_curves, make a calibration plot
        that contains all sets (train/val/test) for each evaluation time.
        """
        xy_lim = 1.01
        # loop over ALL models
        for model_index, calib_model in enumerate(self.calib_curves):
            scaler, selector, model, search, train_strategy = self.combs[model_index]
            for i, time_name in enumerate(self.eval_times_names):
                fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 6), gridspec_kw={'height_ratios': [3, 1]},
                    sharex=True)
                ax1, ax2 = ax
                # identity line
                ax1.plot([0, xy_lim], [0, xy_lim], color="darkgrey", linewidth=0.8)
                # for each set (train/val/test)
                for set_idx, set_ in enumerate(self.set_names):
                    calib_curve_t = calib_model[set_][time_name]
                    # use a clean palette for sets
                    color = self.colors[set_idx]
                    label = self.sets_label_dict[set_]
                    # reuse the central helper
                    self._plot_single_calibration_curve(
                        ax1,
                        ax2,
                        calib_curve_t,
                        label,
                        color,
                        lw_based_on_density,
                    )
                # formatting
                ax1.set_xlim(-0.01, xy_lim)
                ax1.set_ylim(-0.01, xy_lim)
                ax1.spines["top"].set_visible(False)
                ax1.spines["right"].set_visible(False)
                ax1.set_ylabel("Observed MACE incidence")
                ax1.set_title(f"{i+1}-year", fontweight="bold")
                ax2.set_xlabel("Predicted MACE risk")
                ax2.set_ylabel("Density")
                leg = ax1.legend()
                for legobj in leg.legend_handles:
                    legobj.set_linewidth(1.5)
                # save plot
                filename = f"calibration_all_sets_{time_name}_{selector}_{model}_{search}_{train_strategy}.{self.plot_format}"
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, filename),
                            dpi=self.dpi,
                            bbox_inches="tight")
                plt.close()

    def decision_curve(self, comb):
        """
        """
        scaler, selector, model, search, train_strategy = comb
        set_names = [s for s in self.set_names if s != "val"]
        for i, set_ in enumerate(set_names):
            # Load risk scores derived from NAC score
            nac_path = f"results/attr/ml/12_02_2026_NAC/risk_at_times_{set_}.xlsx"
            df_nac = pd.read_excel(nac_path)
            df_nac.columns = [f"{c}_nac" for c in df_nac.columns]
            i_set = self.suffixes_test.index(set_)
            df = pd.DataFrame(self.risk_times_tests[i_set], columns=self.eval_times_names)
            df["outcome"] = self.y_tests[i_set][self.event_column]
            df["time_to_outcome"] = self.y_tests[i_set][self.time_column]
            df = pd.concat([df, df_nac], axis=1)
            x_maxs = [0.51, 0.77, 0.85]
            y_maxs = [0.136, 0.249, 0.396]
            for j, time_name in enumerate(self.eval_times_names):
                filename = f"decision_curve_{time_name}_{selector}_{model}_{search}_{train_strategy}_{set_}_{time_name}.{self.plot_format}"

                df_dca = dca(
                    data=df,
                    outcome="outcome",
                    modelnames=[self.eval_times_names[j]],#, f"{self.eval_times_names[j]}_nac"],
                    thresholds=np.arange(0, x_maxs[j], 0.01),
                    time=self.eval_times[j],
                    time_to_outcome_col="time_to_outcome",
                )
                df_dca["model"] = df_dca["model"].replace({f"{self.eval_times_names[j]}_nac": "NAC score", self.eval_times_names[j]: "Model", "all": "Intervention for all", "none": "Intervention for none"})
                # logger.info(df_dca[df_dca["model"] == "Model"]["net_benefit"].max())
                max_ = y_maxs[j]
                max_ = max_ + max_ * 0.05
                min_ = -max_ * 0.25
                plot_graphs(
                    plot_df=df_dca,
                    graph_type="net_benefit",
                    y_limits=[min_, max_],
                    smooth_frac=0.8,
                    color_names=["#33a02c", "#1f78b4", "#e31a1c"],
                    dpi=600,
                    linewidths=[2, 2, 2],
                    file_name=os.path.join(self.out_dir, filename),
                    show_grid=True
                )
                plt.title(f"{self.sets_label_dict[set_]}", fontweight="bold")
                plt.close()

    def to_structured_array(self, df):
        return np.array(
            list(zip(df[self.event_column], df[self.time_column])),
            dtype=[(self.event_column, '?'), (self.time_column, '<f8')],
        )
