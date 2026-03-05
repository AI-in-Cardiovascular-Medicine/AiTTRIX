import os
import pickle

import numpy as np
import pandas as pd
import copy

from loguru import logger
from sklearn.experimental import enable_iterative_imputer  # required for IterativeImputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from skmultilearn.model_selection import iterative_train_test_split
from lifelines.utils import concordance_index
# from hyperimpute.plugins.imputers import Imputers
from missforest import MissForest
from lifelines import CoxPHFitter
pd.options.mode.chained_assignment = None
from neuroCombat import neuroCombat

from survival.utils import df_to_structured_array
from preprocessing.cleaning_utils import get_continuous_categorical_features

class Preprocessing:
    def __init__(self, config) -> None:
        self.train_file = config.meta.train_file
        self.test_files = config.meta.test_files
        self.suffixes_test = config.meta.suffixes_test
        if isinstance(self.test_files, str):
            self.test_files = [self.test_files]
        self.event_column = config.meta.events
        self.time_column = config.meta.times
        self.sex_column = config.meta.sex_column
        self.pat_id_column = config.meta.pat_id_column
        self.columns_to_drop = config.preprocessing.columns_to_drop
        self.save_as_pickle = config.preprocessing.save_as_pickle
        self.corr_threshold = config.preprocessing.corr_threshold
        self.test_size = config.preprocessing.test_size
        self.replace_zero_time_with = config.preprocessing.replace_zero_time_with
        self.clip_imputed_values = config.preprocessing.clip_imputed_values
        self.out_dir = config.meta.out_dir
        # Initialize competing-event labels to None
        self.comp_event_label_train, self.pat_id_train = None, None
        self.comp_event_label_test = []
        self.x_train, self.y_train, self.x_tests, self.y_tests = None, None, [], []  # initialize
        self.imputers_test = []
        self.raw_train, self.raw_tests = None, []

    def __call__(self, seed):
        self.seed = seed
        os.makedirs(self.out_dir, exist_ok=True)

        # Check if the imputed data file already exists
        data_out_file = os.path.join(self.out_dir, "data_imputed.pkl")
        if os.path.exists(data_out_file):
            logger.info(f"Found existing imputed data file at {data_out_file}, loading data.")
            with open(data_out_file, 'rb') as f:
                data_dict = pickle.load(f)
            # Assign the loaded data directly
            self.x_train = data_dict["x_train"]
            self.x_tests = data_dict["x_tests"]
            self.y_train = data_dict["y_train"]
            self.y_tests = data_dict["y_tests"]
            self.pat_id_train = data_dict.get("pat_id_train", None)
        else:
            self.load_data()  # load data
            if self.test_files is not None:
                self.load_test()
            else:
                self.split_data()  # split if test is not provided
            self.impute_data()  # impute data
            self.univariate_analysis()
            if self.corr_threshold is not None:
                self.remove_highly_correlated_features()
            else:
                self.correlated_to_drop = []
            if self.save_as_pickle:
                self.save()
        # scikit-survival requires structured array
        self.y_train = df_to_structured_array(self.y_train, self.event_column, self.time_column)
        for i, y_test in enumerate(self.y_tests):
            self.y_tests[i] = df_to_structured_array(y_test, self.event_column, self.time_column)
        logger.info(f"Data loaded and preprocessed. Number of features: {self.x_train.shape[1]}, "
                    f"Train patients: {len(self.y_train)}, Test patients: {[len(x_test) for x_test in self.x_tests]}")
        return self.x_train, self.x_tests, self.y_train, self.y_tests

    def load_data(self):
        try:
            data = pd.read_excel(self.train_file)
        except FileNotFoundError:
            logger.error(f'File {self.train_file} not found, check the path in the config.yaml file.')
            raise
        data = data.apply(pd.to_numeric, errors='coerce')  # replace non-numeric entries with NaN
        data = data.dropna(how='all', axis=1)  # drop columns with all NaN
        data.columns = [col.replace(" ", "_") for col in data.columns]
        self.train_raw = data
        # Drop columns from predictors
        cols_to_drop = [self.time_column, self.event_column]
        if self.columns_to_drop is not None and len(self.columns_to_drop) > 0:
            cols_to_drop += self.columns_to_drop
            logger.info(f"Dropping features {self.columns_to_drop}.")
        self.data_x = data.drop(columns=cols_to_drop)
        if self.pat_id_column is not None:
            self.data_x = self.data_x.drop(columns=[self.pat_id_column])
            self.pat_id = data[self.pat_id_column]
        self.data_y = data[[self.event_column, self.time_column]]
        self.data_y[self.time_column] = self.data_y[self.time_column].replace(0, self.replace_zero_time_with)  # some models do not accept t <= 0 -> set to small value > 0


    def _check_test_columns(self, test_data, test_file):
        train_cols = set(self.x_train.columns)
        test_cols = set(test_data.columns)
        if train_cols != test_cols:
            missing_in_test = train_cols - test_cols
            extra_in_test = test_cols - train_cols
            msg = (
                f"Columns in test file {test_file} do not match those in the train file.\n"
                f"Missing in test: {sorted(missing_in_test)}\n"
                f"Extra in test: {sorted(extra_in_test)}"
            )
            logger.error(msg)
            raise ValueError(msg)


    def load_test(self):
        """If external test data is provided, load it (no train-test split)."""
        self.x_train = self.data_x
        if self.pat_id_column is not None:
            self.pat_id_train = self.pat_id
        else:
            self.pat_id_train = None
        self.y_train = self.data_y
        for test_file in self.test_files:
            try:
                data = pd.read_excel(test_file)
            except FileNotFoundError:
                logger.error(f'File {test_file} not found, check the path in the config.yaml file.')
                raise
            data = data.apply(pd.to_numeric, errors='coerce')  # replace non-numeric entries with NaN
            data = data.dropna(how='all', axis=1)  # drop columns with all NaN
            data.columns = [col.replace(" ", "_") for col in data.columns]
            self.raw_tests.append(data)
            # Drop columns from predictors
            cols_to_drop = [self.time_column, self.event_column]
            if self.columns_to_drop is not None and len(self.columns_to_drop) > 0:
                cols_to_drop += self.columns_to_drop
            x_test = data.drop(columns=cols_to_drop)
            self._check_test_columns(x_test, test_file)  # Check if columns match
            x_test = x_test[self.x_train.columns]  # Order test columns as in train
            y_test = data[[self.event_column, self.time_column]]
            y_test[self.time_column] = y_test[self.time_column].replace(0, self.replace_zero_time_with)  # some models don't accept t=0 -> set to small value > 0
            self.x_tests.append(x_test)
            self.y_tests.append(y_test)

    def split_data(self):
        """Train-test split stratified by outcome and censoring time. Apply only if the external test set is not given"""
        cuts = np.linspace(self.data_y[self.time_column].min(), self.data_y[self.time_column].max(), num=10)
        durations_discrete = np.searchsorted(cuts, self.data_y[self.time_column], side='left')
        y = np.array([(event, duration) for event, duration in zip(self.data_y[self.event_column], durations_discrete)])
        idx_all = np.expand_dims(np.arange(len(y), dtype=int), axis=1)
        idx_train, _, idx_test, _ = iterative_train_test_split(idx_all, y, test_size=self.test_size)
        self.x_train = self.data_x.iloc[idx_train[:, 0]]
        self.y_train = self.data_y.iloc[idx_train[:, 0]]
        self.x_tests = [self.data_x.iloc[idx_test[:, 0]]]
        self.y_tests = [self.data_y.iloc[idx_test[:, 0]]]

    def _create_imputer(self, categorical=None):
        """
        Factory method to create an imputer instance. Returns a fresh imputer, ready to fit.
        """
        rsf_clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        rsf_rgr = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        return MissForest(clf=rsf_clf, rgr=rsf_rgr, categorical=categorical, max_iter=5, verbose=1)

    def impute_data(self):
        _, categorical = get_continuous_categorical_features(self.x_train, separate_binary=False)
        categorical = None if len(categorical) == 0 else categorical
        logger.info(f"Imputing data with MissForest")
        self.imputer_train = self._create_imputer(categorical=categorical)  # Create training imputer
        if self.x_train.isna().sum().sum() > 0:
            self.imputer_train.fit(self.x_train)
            imp_train = self.imputer_train.transform(self.x_train)
            self.x_train = pd.DataFrame(imp_train, index=self.x_train.index, columns=self.x_train.columns)
            if self.clip_imputed_values:
                for col in self.x_train.columns:
                    min_val = self.x_train[col].min()
                    max_val = self.x_train[col].max()
                    self.x_train[col] = self.x_train[col].clip(lower=min_val, upper=max_val)

        for i, x_test in enumerate(self.x_tests):
            x_test = x_test[self.x_train.columns]
            if x_test.isna().sum().sum() > 0:
                x_test_imp = self.imputer_train.transform(x_test)
                logger.info(f"   Imputed missing values in test set '{self.suffixes_test[i]}'")
                x_test_imp = pd.DataFrame(x_test_imp, index=x_test.index, columns=x_test.columns)
                # Clip test set to train min/max
                if self.clip_imputed_values:
                    for col in x_test_imp.columns:
                        min_val = self.x_train[col].min()
                        max_val = self.x_train[col].max()
                        x_test_imp[col] = x_test_imp[col].clip(lower=min_val, upper=max_val)
                self.x_tests[i] = x_test_imp
        if self.clip_imputed_values:
            logger.info(f"   Imputed values clipped to min/max.")

    def univariate_analysis(self):
        features_univariate = []
        results = []
        for feature in self.x_train.columns:
            try:
                # --- Train Cox on training data ---
                df_train = pd.concat([self.x_train[[feature]], self.y_train], axis=1)
                cph = CoxPHFitter()
                cph.fit(df_train, duration_col=self.time_column, event_col=self.event_column, formula=feature)
                # Extract training metrics
                hr_low = cph.summary["exp(coef) lower 95%"][feature]
                hr_up = cph.summary["exp(coef) upper 95%"][feature]
                hr = cph.summary["exp(coef)"][feature]
                pval = cph.summary["p"][feature]
                c_index_train = cph.concordance_index_
                # Feature selection
                if c_index_train > 0.55 or hr_low > 1 or hr_up < 1 or pval < 0.05:
                    features_univariate.append(feature)
                # --- Evaluate on test data ---
                res_dict = {
                    "feature": feature,
                    "HR": hr,
                    "HR_lower_95": hr_low,
                    "HR_upper_95": hr_up,
                    "p_value": pval,
                    "Cindex_train": c_index_train,
                }
                for x_test, y_test, test_suffix in zip(self.x_tests, self.y_tests, self.suffixes_test):
                    df_test = pd.concat([x_test[[feature]], y_test], axis=1)
                    risk_scores_test = cph.predict_partial_hazard(df_test)
                    c_index_test = concordance_index(df_test[self.time_column],
                                                     -risk_scores_test.values.flatten(),
                                                     df_test[self.event_column])
                    res_dict.update({f"Cindex_{test_suffix}": c_index_test})
                # Store results
                results.append(res_dict)
            except Exception as e:
                print(f"Feature {feature}: error encountered. {str(e)}")
        # Apply selection
        if self.sex_column is not None and self.sex_column not in features_univariate:
            features_univariate.append(self.sex_column)
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_excel(os.path.join(self.out_dir, "univariate_results_train_test.xlsx"), index=False)

    def remove_highly_correlated_features(self):

        logger.info(f"Removing highly correlated features with threshold {self.corr_threshold}")
        corr_matrix = self.x_train.corr()  # correlation matrix
        # compute importance based on c-index in cox univariate model
        c_index = {}
        for feature in self.x_train.columns:
            print(feature)
            df_cox = pd.concat([self.x_train[feature], self.y_train], axis=1)
            cph = CoxPHFitter()
            cph.fit(df_cox, duration_col=self.time_column, event_col=self.event_column, formula=feature, fit_options={"step_size": 0.1})
            c_index[feature] = cph.concordance_index_
        importance = pd.Series(c_index).sort_values(ascending=False)
        # create upper triangle matrix and order by c-index. Then loop over features from the most important to the
        # least important and remove a feature if highly correlated to another one with higher c-index
        corr_matrix = corr_matrix.reindex(index=importance.index, columns=importance.index).abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.correlated_to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > self.corr_threshold)]
        if len(self.correlated_to_drop) > 0:
            logger.info(f"{len(self.correlated_to_drop)} highly correlated features were removed:\n {self.correlated_to_drop}")
        self.x_train = self.x_train.drop(columns=self.correlated_to_drop)
        for i, x_test in enumerate(self.x_tests):
            self.x_tests[i] = x_test.drop(columns=self.correlated_to_drop)

    def save(self):
        # Save imputed data
        data_dict = {
            "x_train": self.x_train,
            "x_tests": self.x_tests,
            "y_train": self.y_train,
            "y_tests": self.y_tests,
            "raw_train": self.raw_train,
            "raw_tests": self.raw_tests,
            "suffixes_test": self.suffixes_test,
            "pat_id_train": self.pat_id_train
        }
        data_out_file = os.path.join(self.out_dir, "data_imputed.pkl")
        with open(data_out_file, 'wb') as f:
            pickle.dump(data_dict, f)
        logger.info(f'Saved imputed data to {data_out_file}')
        # Save preprocessing pipeline
        preprocessing_steps = {
            "imputer_train": self.imputer_train,
            "imputers_test": self.imputers_test,
            "correlation_features_to_drop": self.correlated_to_drop
        }
        with open(os.path.join(self.out_dir, "preprocessing_steps.pkl"), "wb") as f:
            pickle.dump(preprocessing_steps, f)