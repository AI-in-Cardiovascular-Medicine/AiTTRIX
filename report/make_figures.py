import os
import pickle
from loguru import logger
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from dcurves import dca, plot_graphs

from evaluation.calibration import calibration_plot_survival

plt.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white"
})

class MakeFigures:
    def __init__(self, config) -> None:
        """Initializes MakeFigures by loading results and preparing all data structures.

        Args:
            config: Configuration object with fields:
                - config.meta.plot_format: Output image format (e.g. 'png', 'pdf').
                - config.meta.times: Name of the time-to-event column.
                - config.meta.events: Name of the event column.
                - config.meta.suffixes_test: List of dataset suffixes for test sets.
                - config.meta.out_dir: Directory containing the results pickle file.
                - config.meta.plot_dpi: DPI for saved figures.
                - config.meta.seed: Random seed used to select results.
        """
        self.plot_format = config.meta.plot_format
        self.time_column = config.meta.times
        self.event_column = config.meta.events
        suffixes_test = config.meta.suffixes_test
        self.set_names = ["val", "swiss", "vienna", "swiss_vienna_treated"]
        self.ids_suffixes_to_keep = [i for i in range(len(suffixes_test)) if suffixes_test[i] in self.set_names]
        self.suffixes_test = [suffixes_test[i] for i in self.ids_suffixes_to_keep]
        self.experiment_dir = config.meta.out_dir
        self.dpi = config.meta.plot_dpi
        self.results_file = os.path.join(self.experiment_dir, 'results.pkl')
        self.seed = config.meta.seed
        self.sets_label_dict = {
            "val": "Validation",
            "vienna": "Vienna Test",
            "swiss": "Swiss Test",
            "swiss_vienna_treated": "Treated",
        }
        self.colors = ["#9A3D3F", '#7FB77E', "#af4b91"]
        n_groups = 3

        with open(self.results_file, 'rb') as f:
            self.results = pickle.load(f)
        results = self.results[self.seed]
        self.out_dir = os.path.join("results", "plots")
        os.makedirs(self.out_dir, exist_ok=True)
        self.x_train = results['x_train']
        self.y_train = results['y_train']
        self.x_tests = [results['x_tests'][i] for i in self.ids_suffixes_to_keep]
        self.y_tests = [results['y_tests'][i] for i in self.ids_suffixes_to_keep]
        self.eval_times = results['eval_times']
        self.eval_times_names = results['eval_times_names']
        self.comb = ("RobustScaler", "SelectKBest", "RSF", "rand", "refit")
        val_ids = np.concat([c[1] for c in results["folds_indices"]])
        self.x_val = self.x_train.iloc[val_ids]
        scaler, selector, model, search, train_strategy = self.comb
        if results[scaler][selector][model][search][train_strategy] == {}:
            msg = f"No trained model for Scaler={scaler}, Selector={selector} , Model={model}, Search={search}."
            logger.error(msg)
            raise ValueError(msg)
        self.df_val_risk = results["val_risk"][scaler][selector][model][search]
        self.best_estimator = results[scaler][selector][model][search][train_strategy]
        # Get risk on TRAIN
        self.risk_time_train = self.get_risk_at_eval_times(self.best_estimator, self.x_train)
        self.risk_train = self.best_estimator.predict(self.x_train)
        # Get Risk scores on TEST
        self.risk_tests = []
        self.risk_times_tests = []
        for i in range(len(self.x_tests)):
            self.risk_tests.append(self.best_estimator.predict(self.x_tests[i]))
            surv_func = self.best_estimator.predict_survival_function(self.x_tests[i])
            self.risk_times_tests.append(np.array([[1 - func(t) for t in self.eval_times] for func in surv_func]))
        # Get Risk scores on VAL
        self.risk_val = results["val_risk"][scaler][selector][model][search]["risk"]
        self.y_val = self.to_structured_array(self.df_val_risk[[self.event_column, self.time_column]])
        self.risk_times_val = self.df_val_risk[self.eval_times_names].values

        # Compute NAC score on val and tests
        self.x_val = self.compute_nac(self.x_val)
        self.x_tests = [self.compute_nac(x_test) for x_test in self.x_tests]

        # Compute grid for risk stratification
        self.grid = np.quantile(self.risk_time_train[:, 2], np.linspace(start=0, stop=1, num=n_groups + 1))
        self.risk_labels = ["Low Risk", "Medium Risk", "High Risk"]
        for i, (set_, risk_scores, risk_scores_time) in enumerate(zip(
            self.set_names[1:],
            self.risk_tests,
            self.risk_times_tests
        )):
            risk_1y = risk_scores_time[:, 2]
            bins = np.digitize(risk_1y, self.grid[1:-1], right=True)

            df = self.x_tests[i]
            df.loc[:, "risk_score"] = risk_1y
            df.loc[:, "risk_class_"] = bins
            df.loc[:, "risk_class"] = df["risk_class_"].map(
                dict(zip(np.arange(n_groups), self.risk_labels))
            )

    def __call__(self):
        """Runs the full figure generation pipeline in sequence."""
        calib_curve = self.calibration_plot()
        self.calib_curve = calib_curve
        self.calibration_plots_by_model()
        self.decision_curve()
        self.km_all_sets()

    @staticmethod
    def compute_nac(df):
        """Computes the NAC (N-terminal pro b-type natriuretic peptide / eGFR) staging score.

        Assigns a NAC stage (1–4) to each row based on NTproBNP and eGFR_CKDEPI values:
        - Stage 4: NTproBNP >= 10 000
        - Stage 1: NTproBNP <= 3 000 and eGFR >= 45
        - Stage 3: NTproBNP > 3 000 and eGFR < 45
        - Stage 2: all remaining valid rows

        Args:
            df: DataFrame containing ``NTproBNP`` and ``eGFR_CKDEPI`` columns.

        Returns:
            pd.DataFrame: Input DataFrame with an added ``NAC_score`` column (float, NaN where
                either input variable is missing).
        """
        df = df.copy()
        df["NAC_score"] = np.nan
        mask_stage4 = df["NTproBNP"] >= 10000
        mask_stage1 = (df["NTproBNP"] <= 3000) & (df["eGFR_CKDEPI"] >= 45)
        mask_stage3 = (df["NTproBNP"] > 3000) & (df["eGFR_CKDEPI"] < 45)
        mask_valid = df["NTproBNP"].notna() & df["eGFR_CKDEPI"].notna()
        df.loc[mask_stage4, "NAC_score"] = 4
        df.loc[mask_stage1 & ~mask_stage4, "NAC_score"] = 1
        df.loc[mask_stage3 & ~mask_stage4, "NAC_score"] = 3
        df.loc[mask_valid & df["NAC_score"].isna(), "NAC_score"] = 2
        return df

    def get_risk_at_eval_times(self, estimator, x):
        """Computes risk scores (1 - survival probability) at predefined evaluation times.

        Args:
            estimator: Trained survival model with a ``predict_survival_function`` method.
            x: Input feature DataFrame.

        Returns:
            np.ndarray: Array of shape (n_samples, n_eval_times) with risk scores.
        """
        surv_func = estimator.predict_survival_function(x)
        return np.array([[1 - func(t) for t in self.eval_times] for func in surv_func])

    def plot_km_stratified_by_risk_nac(self, evaluation_set):
        """Plots side-by-side Kaplan-Meier curves stratified by ML risk class and NAC stage.

        Saves the figure to disk and logs survival statistics and pairwise log-rank
        test results for both stratification schemes.

        Args:
            evaluation_set: Dataset suffix identifying the test set to plot
                (e.g. ``'swiss'``, ``'vienna'``, ``'swiss_vienna_treated'``).
        """
        logger.info(f"==== {evaluation_set} ====")
        id_test_combined = np.nonzero(np.array(self.suffixes_test) == evaluation_set)[0][0]
        x_test = self.x_tests[id_test_combined].reset_index(drop=True).copy()
        y_test = pd.DataFrame(self.y_tests[id_test_combined]).reset_index(drop=True)
        x = pd.concat([x_test, y_test], axis=1)

        # ML risk classes and NAC groups
        ml_order = [
            "Low Risk",
            "Medium Risk",
            "High Risk",
        ]
        map_center = {
            'swiss': "Swiss test",
            'vienna': "Vienna test",
            'swiss_vienna_treated': "Treated"
        }
        nac_order = ["NAC I", "NAC II", "NAC III/IV"]

        # Color palettes
        ml_colors = {
            "Low Risk": "#2E8B57",
            "Medium Risk": "#FF8C42",
            "High Risk": "#E65047"
        }
        nac_colors = {
            "NAC I": "#6BAED6",
            "NAC II": "#3182BD",
            "NAC III/IV": "#756BB1",
        }

        # NAC GROUPING (merge stages 3–4)
        x["NAC_group"] = x["NAC_score"].apply(
            lambda v: "NAC I" if v == 1 else
            "NAC II" if v == 2 else
            "NAC III/IV"
        )

        fig, (ax_ml, ax_nac) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

        # LEFT PANEL — ML RISK CLASSES
        km_ml_objects = {}
        for label in ml_order:
            subset = x[x["risk_class"] == label]
            km = KaplanMeierFitter()
            km.fit(
                durations=subset["time_to_first_mace"].astype(float) / 365.25,
                event_observed=subset["mace"].astype(int),
                label=label.replace(" Risk", "")
            )
            km.plot_survival_function(
                ax=ax_ml,
                ci_show=False,
                loc=slice(0, 3 * 1.03),
                color=ml_colors[label],
                linewidth=3
            )
            km_ml_objects[label] = km
        add_at_risk_counts(*km_ml_objects.values(), ax=ax_ml, rows_to_show=["At risk"], fontsize=13)
        ax_ml.set_title(f"ML Risk Classes - {map_center[evaluation_set]}", fontsize=16, fontweight='bold')
        ax_ml.set_xlabel("Years", fontsize=15)
        ax_ml.set_ylabel("MACE-free survival", fontsize=15)
        ax_ml.tick_params(axis='x', labelsize=14)
        ax_ml.tick_params(axis='y', labelsize=14)
        ax_ml.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax_ml.yaxis.grid(True, color='gray', alpha=0.2, linestyle='-')
        ax_ml.spines['top'].set_visible(False)
        ax_ml.spines['right'].set_visible(False)

        # RIGHT PANEL — NAC GROUPS
        km_nac_objects = {}
        for label in nac_order:
            subset = x[x["NAC_group"] == label]
            km = KaplanMeierFitter()
            km.fit(
                durations=subset["time_to_first_mace"].astype(float) / 365.25,
                event_observed=subset["mace"].astype(int),
                label=label
            )
            km.plot_survival_function(
                ax=ax_nac,
                ci_show=False,
                loc=slice(0, 3 * 1.03),
                color=nac_colors[label],
                linewidth=3
            )
            km_nac_objects[label] = km
        add_at_risk_counts(*km_nac_objects.values(), ax=ax_nac, rows_to_show=["At risk"], fontsize=13)
        ax_nac.set_title(f"NAC Staging System - {map_center[evaluation_set]}", fontsize=16, fontweight='bold')
        ax_nac.set_xlabel("Years", fontsize=15)
        ax_nac.tick_params(labelleft=True)
        ax_nac.tick_params(axis='x', labelsize=14)
        ax_nac.tick_params(axis='y', labelsize=14)
        ax_nac.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax_nac.yaxis.grid(True, color='gray', alpha=0.2, linestyle='-')
        ax_nac.spines['top'].set_visible(False)
        ax_nac.spines['right'].set_visible(False)

        plt.ylim(0.24, 1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, f"km_stratified_by_risk_nac_{evaluation_set}.png"), dpi=600)
        plt.close()

        # Log survival statistics and pairwise log-rank tests
        def print_km_survival(order, km_objects, subset_col, event_col, eval_time_points, group_label):
            logger.info(f"--- {group_label} ---")
            for group in order:
                grp_subset = x[x[subset_col] == group]
                km_fitter = km_objects[group]
                surv = km_fitter.survival_function_at_times(eval_time_points)
                n = len(grp_subset)
                n_events = grp_subset[event_col].sum()
                logger.info(f"{group}")
                logger.info(f"  N = {n}")
                logger.info(f"  Events = {int(n_events)} ({n_events / n * 100:.1f}%)")
                for t, s in zip(eval_time_points, surv.values):
                    logger.info(f"  KM survival at {t} year(s): {s:.3f}")

        def print_pairwise_logrank(order, subset_col, group_label, time_col="time_to_first_mace", event_col="mace"):
            logger.info(f"Pairwise log-rank tests for {group_label} groups:")
            for g1, g2 in combinations(order, 2):
                df1 = x[x[subset_col] == g1]
                df2 = x[x[subset_col] == g2]
                if df1.empty or df2.empty:
                    logger.info(f"Skipping {g1} vs {g2} (empty group)")
                    continue
                result = logrank_test(
                    df1[time_col], df2[time_col],
                    event_observed_A=df1[event_col],
                    event_observed_B=df2[event_col]
                )
                logger.info(f"{g1} vs {g2}: p-value = {result.p_value:.4f}, test statistic = {result.test_statistic:.2f}")

        time_points = [1, 2, 3]
        print_km_survival(ml_order, km_ml_objects, "risk_class", "mace", time_points, "ML")
        print_km_survival(nac_order, km_nac_objects, "NAC_group", "mace", time_points, "NAC")
        print_pairwise_logrank(ml_order, "risk_class", "ML")
        print_pairwise_logrank(nac_order, "NAC_group", "NAC")

    def km_all_sets(self):
        """Generates KM stratification plots for every test set."""
        for evaluation_set in self.suffixes_test:
            self.plot_km_stratified_by_risk_nac(evaluation_set)

    def calibration_plot(self):
        """Computes calibration curves for all sets and evaluation times.

        Returns:
            dict: Nested dict ``{set_name: {time_name: {"grid", "predict_grid", "risk"}}}``.
        """
        calib_curves = {set_name: {} for set_name in self.set_names}
        for set_, risk_scores, y in zip(self.set_names,
                                        [self.risk_times_val] + self.risk_times_tests,
                                        [self.y_val] + self.y_tests):
            for i, time_name in enumerate(self.eval_times_names):
                fig, grid, predict_grid, risk = calibration_plot_survival(durations=y[self.time_column],
                                                                          events=y[self.event_column],
                                                                          risk=risk_scores[:, i],
                                                                          time=self.eval_times[i])
                calib_curves[set_][time_name] = {
                    "grid": grid,
                    "predict_grid": predict_grid,
                    "risk": risk
                }
        return calib_curves

    @staticmethod
    def _plot_single_calibration_curve(ax1, ax2, calib_curve_t, label, color, lw_based_on_density=True):
        """Plots a single calibration curve and its risk density onto given axes.

        Args:
            ax1: Matplotlib ``Axes`` for the calibration curve (upper panel).
            ax2: Matplotlib ``Axes`` for the KDE density plot (lower panel).
            calib_curve_t: Dict with keys ``'grid'``, ``'predict_grid'``, and ``'risk'``
                as returned by ``calibration_plot_survival``.
            label: Legend label for this curve.
            color: Line color.
            lw_based_on_density: If ``True``, line width varies with risk density via KDE.
        """
        if lw_based_on_density:
            x_orig = calib_curve_t["grid"]
            y_orig = calib_curve_t["predict_grid"]
            risk = calib_curve_t["risk"]
            x = np.linspace(x_orig.min(), x_orig.max(), 500)
            interp_fn = interp1d(x_orig, y_orig, kind="cubic")
            y = interp_fn(x)
            kde = gaussian_kde(risk, bw_method=0.3)
            density = kde.evaluate(x)
            lw = 0.9 + 2.4 * (density - density.min()) / (density.max() - density.min())
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, linewidths=lw[:-1], color=color, label=label)
            ax1.add_collection(lc)
        else:
            ax1.plot(calib_curve_t["grid"], calib_curve_t["predict_grid"],
                     "-", linewidth=2, color=color, label=label)
        sns.kdeplot(calib_curve_t["risk"], ax=ax2, color=color, bw_adjust=0.5, fill=True, alpha=0.15)

    def calibration_plots_by_model(self, lw_based_on_density=True):
        """Saves one calibration figure per evaluation time, overlaying all test sets.

        Args:
            lw_based_on_density: Passed through to ``_plot_single_calibration_curve``
                to control variable-width line rendering.
        """
        xy_lim = 1.01
        scaler, selector, model, search, train_strategy = self.comb
        for i, time_name in enumerate(self.eval_times_names):
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 6), gridspec_kw={'height_ratios': [3, 1]},
                sharex=True)
            ax1, ax2 = ax
            ax1.plot([0, xy_lim], [0, xy_lim], color="darkgrey", linewidth=0.8)
            for set_idx, set_ in enumerate(["swiss", "vienna"]):
                calib_curve_t = self.calib_curve[set_][time_name]
                self._plot_single_calibration_curve(
                    ax1, ax2, calib_curve_t,
                    label=self.sets_label_dict[set_],
                    color=self.colors[set_idx],
                    lw_based_on_density=lw_based_on_density,
                )
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
            filename = f"calibration_all_sets_{time_name}_{selector}_{model}_{search}_{train_strategy}.{self.plot_format}"
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, filename), dpi=self.dpi, bbox_inches="tight")
            plt.close()

    def decision_curve(self):
        """Generates and saves decision curve analysis plots for each test set and evaluation time."""
        set_names = [s for s in self.set_names if s != "val"]
        x_maxs = [0.51, 0.77, 0.85]
        y_maxs = [0.136, 0.249, 0.396]
        for i, set_ in enumerate(set_names):
            i_set = self.suffixes_test.index(set_)
            df = pd.DataFrame(self.risk_times_tests[i_set], columns=self.eval_times_names)
            df["outcome"] = self.y_tests[i_set][self.event_column]
            df["time_to_outcome"] = self.y_tests[i_set][self.time_column]
            for j, time_name in enumerate(self.eval_times_names):
                filename = f"decision_curve_{set_}_{time_name}.{self.plot_format}"
                df_dca = dca(
                    data=df,
                    outcome="outcome",
                    modelnames=[self.eval_times_names[j]],
                    thresholds=np.arange(0, x_maxs[j], 0.01),
                    time=self.eval_times[j],
                    time_to_outcome_col="time_to_outcome",
                )
                df_dca["model"] = df_dca["model"].replace({
                    self.eval_times_names[j]: "Model",
                    "all": "Intervention for all",
                    "none": "Intervention for none",
                })
                max_ = y_maxs[j] * 1.05
                plot_graphs(
                    plot_df=df_dca,
                    graph_type="net_benefit",
                    y_limits=[-max_ * 0.25, max_],
                    smooth_frac=0.8,
                    color_names=["#33a02c", "#1f78b4", "#e31a1c"],
                    dpi=self.dpi,
                    linewidths=[2, 2, 2],
                    file_name=os.path.join(self.out_dir, filename),
                    show_grid=True
                )
                plt.title(f"{self.sets_label_dict[set_]}", fontweight="bold")
                plt.close()

    def to_structured_array(self, df):
        """Converts a DataFrame with event and time columns to a NumPy structured array.

        Args:
            df: DataFrame containing ``self.event_column`` (bool) and
                ``self.time_column`` (float) columns.

        Returns:
            np.ndarray: Structured array with dtype
                ``[(event_column, '?'), (time_column, '<f8')]``.
        """
        return np.array(
            list(zip(df[self.event_column], df[self.time_column])),
            dtype=[(self.event_column, '?'), (self.time_column, '<f8')],
        )