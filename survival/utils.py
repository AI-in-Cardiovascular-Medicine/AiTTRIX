import numpy as np
import pandas as pd
from sksurv.functions import StepFunction
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold


def _array_to_step_function(time_points, array):
    """
    Convert an array of survival values into an array of step functions.

    Args:
        time_points (array-like): Time points defining the x-axis of the step function.
        array (np.ndarray): 2D array of shape (n_samples, n_time_points) with survival values.

    Returns:
        np.ndarray: 1D array of StepFunction objects, one per sample.
    """
    n_samples = array.shape[0]
    funcs = np.empty(n_samples, dtype=np.object_)
    for i in range(n_samples):
        funcs[i] = StepFunction(x=time_points, y=array[i])
    return funcs


def df_to_structured_array(df: pd.DataFrame, event_column: str, time_column: str):
    """
    Convert a DataFrame to a NumPy structured array for survival analysis.

    Args:
        df (pd.DataFrame): Input DataFrame containing event and time columns.
        event_column (str): Name of the binary event indicator column.
        time_column (str): Name of the time-to-event column.

    Returns:
        np.ndarray: Structured array with boolean event and float time fields.
    """
    return np.array(
        list(zip(df[event_column], df[time_column])),
        dtype=[(event_column, '?'), (time_column, '<f8')],
    )


def stratified_split(y, time_col, event_col, n_splits, seed, n_bins=5, groups=None):
    """
    Create stratified cross-validation splits for survival data based on discretized event times, preserving the
    distribution of events and censoring across folds.

    Parameters:
        y (np.ndarray): Structured NumPy array with fields for time and event.
        time_col (str): Name of the field containing time to event/censoring.
        event_col (str): Name of the field indicating event occurrence (1 if event, 0 if censored).
        n_splits (int): Number of stratified folds.
        seed (int): Random seed for reproducibility.
        n_bins (int, optional): Number of bins to discretize event times (default is 5).
        groups (np.ndarray, optional): Group labels for samples to ensure group-wise splitting (default is None).

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: List of (train_idx, test_idx) splits.
    """
    event_times = y[time_col][y[event_col] == 1]  # get event times
    bins = np.linspace(event_times.min(), event_times.max(), num=n_bins)  # Discretize durations
    discrete_labels = np.searchsorted(bins, y[time_col], side='left')
    discrete_labels = np.where(y[event_col], discrete_labels, 0)  # Assign 0 where no event occurred
    if groups is None:
        cv = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
        folds = [x for x in cv.split(np.zeros(len(y)), discrete_labels)]
    else:
        cv = StratifiedGroupKFold(n_splits=n_splits, random_state=seed, shuffle=True)
        folds = [x for x in cv.split(np.zeros(len(y)), discrete_labels, groups)]
    return folds
