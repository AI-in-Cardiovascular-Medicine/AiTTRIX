import pandas as pd
import re
import os
from os.path import join
import numpy as np
import json
from tableone import TableOne
import pickle as pkl

base_path = "results/"
OUT_FOLDER = join(base_path, "tables")
os.makedirs(OUT_FOLDER, exist_ok=True)

splits = {
    "val": "Internal Validation",
    "swiss": "Swiss Test",
    "vienna": "Vienna Test",
    "swiss_vienna_treated": "Test sets combined (treated)",
}

with open("datasets/features_dict_no_parenthesis.json") as f:
    features_mapping = json.load(f)

def table1():
    def order_amyloidosis_subtype(table):
        amyloidosis_order = [
            "TTR wild type",
            "TTR unspecified",
            "TTR h/m",
            "TTR-AL",
            "AL",
            "Unknown",
        ]
        idx = table.index
        amy_idx = [i for i in idx if i[0] == "Amyloidosis Subtype"]
        other_idx = [i for i in idx if i[0] != "Amyloidosis Subtype"]
        amy_idx_sorted = sorted(
            amy_idx,
            key=lambda x: amyloidosis_order.index(x[1]) if x[1] in amyloidosis_order else 999
        )
        new_index = pd.MultiIndex.from_tuples(amy_idx_sorted + other_idx)
        table = table.reindex(new_index)
        return table
    desired_order = [
        "Amyloidosis Subtype",
        "Age at Baseline",
        "Gender",
        "BMI (kg/m²)",
        "NYHA Class",
        "NAC Score",
        "NT-proBNP (pg/mL)",
        "Creatinine (µmol/L)",
        "eGFR (CKD-EPI, mL/min/1.73m²)",
        "Potassium (mmol/L)",
        "Sodium (mmol/L)",
        "Hemoglobin (g/dL)",
        "LVEF (%)",
        "LVEDD (mm)",
        "Maximum Wall Thickness (mm)",
        "History of Valvular Heart Disease",
        "Aortic Valve Disease > Grade I",
        "Mitral Valve Disease > Grade I",
        "Tricuspid Valve Disease > Grade I",
        "AF / AFL",
        "Pacemaker Implanted",
        "Arterial Hypertension",
        "Diabetes Mellitus",
        "History of Cancer",
        "Amyloid-specific Therapy",
        "Beta-blocker Therapy",
        "RAAS Inhibitor Therapy",
        "MRA Therapy",
        "SGLT2 Inhibitor Therapy",
        "Diuretic Therapy",
    ]
    train = pd.read_excel("datasets/train.xlsx")
    swiss = pd.read_excel("datasets/swiss.xlsx")
    vienna = pd.read_excel("datasets/vienna.xlsx")
    train["Set"] = "Training Set"
    swiss["Set"] = "Swiss Test Set"
    vienna["Set"] = "Vienna Test Set"
    vienna["Center"] = "Vienna"
    df = pd.concat([train, swiss, vienna])
    df_complete = df.copy()
    df = df.drop(columns=["mace", "time_to_first_mace", "Record_ID"])
    df = df.reset_index(drop=True)
    count_unique = df.nunique()
    categorical_features = count_unique[
        (count_unique > 2) & (count_unique < 10)].index.tolist()  # Features with less than 5 unique values
    binary_features = count_unique[count_unique == 2].index.tolist()  # Features with less than 5 unique values
    categorical_features.remove("Set")
    df[binary_features] = (
        df[binary_features]
        .astype('Int64')  # nullable int → preserves NA
        .astype('string')  # convert to string, NA stays NA
    )
    continuous_features = count_unique[count_unique >= 10].index.tolist()  # Features with more than 5 unique values
    cols_to_show = df.columns.values.tolist()
    order = {f: ["1", "0"] for f in binary_features}
    limit = {col: 1 for col in binary_features}
    decimals = {col: 1 for col in binary_features + categorical_features}
    decimals.update({col: 0 for col in continuous_features})
    table = TableOne(df, columns=cols_to_show, nonnormal=continuous_features,
                     categorical=categorical_features + binary_features, groupby="Set", pval=True, order=order,
                     limit=limit, missing=False, label_suffix=False, decimals=decimals, htest_name=True)
    df_table = table.tableone
    df_table.index = df_table.index.set_levels(
        df_table.index.levels[0].to_series().replace(features_mapping),
        level=0
    )
    df_table = df_table.reindex(desired_order, level=0)
    df_table.columns = df_table.columns.get_level_values(1)
    out_path = join(OUT_FOLDER, "table1_by_set.xlsx")
    df_table = df_table[["Overall", "Training Set", "Swiss Test Set", "Vienna Test Set", "P-Value"]]
    df_table = order_amyloidosis_subtype(df_table)
    df_table.to_excel(out_path, index=True)
    return df_complete

def _missing_summary(x):
    n_missing = x.isna().sum()
    pct = n_missing / len(x) * 100
    return f"{n_missing} ({pct:.1f})"

def missing_values_count(df):
    with open("results/rsf/results.pkl", "rb") as f:
        results = pkl.load(f)
    train_features = results[42]["x_train"].columns.values.tolist()
    df = df[train_features + ["Set"]]
    df_nan = df.groupby("Set").apply(lambda group: group.apply(_missing_summary), include_groups=False).T
    df_nan_overall = pd.DataFrame(df.apply(_missing_summary))
    df_nan_overall.columns = ["Overall"]
    df_nan = pd.concat([df_nan, df_nan_overall], axis=1)
    # index renaming
    df_nan.index = df_nan.index.map(features_mapping)
    df_nan = df_nan[['Overall', 'Training Set', 'Swiss Test Set', 'Vienna Test Set']]
    order = [c for c in df.drop(columns=["Set"]).isna().sum().sort_values(ascending=False).index.map(features_mapping)]
    df_nan = df_nan.loc[order]
    out_path = join(OUT_FOLDER, "missing_values_by_set.xlsx")
    df_nan.to_excel(out_path, index=True)
    return

def split_metric_name(col_name):
    """
    Convert single-level metric name to (family, label) tuple
    """
    # C-index metrics
    if col_name.startswith(("Harrell", "Uno", "Antolini")):
        return ("C-index", col_name.replace(" (95% CI)", ""))
    # AUC metrics
    m = re.match(r"AUC \((\d+)y\)", col_name)
    if m:
        year = int(m.group(1))
        return ("AUC", f"{year} Year")
    # ICI metrics
    m = re.match(r"ICI \((\d+)y\)", col_name)
    if m:
        year = int(m.group(1))
        return ("ICI", f"{year} Year")
    # Mean Calibration metrics
    m = re.match(r"Mean Calibration \((\d+)y\)", col_name)
    if m:
        year = int(m.group(1))
        return ("Mean Calibration", f"{year} Year")
    # fallback
    return ("Other", col_name)

def round_ci(s):
    down, up = s.replace("]", "").replace("[", "").replace(" ", "").split(",")
    s = f"[{np.round(float(down), 2)}, {np.round(float(up), 2)}]"
    return s

def discrimination_table(res):
    data = res.copy()
    # --- metrics names mapping ---
    discrimination_metrics_mapping = {
        "harrell": "Harrell",
        "uno": "Uno",
        "ant": "Antolini",
    }
    auc_times = ["1y", "2y", "3y"]
    for t in auc_times:
        discrimination_metrics_mapping[f"auc_{t}"] = f"AUC ({t})"
    # --- columns to extract and map ---
    id_vars = ["Model"]
    value_cols = []
    metric_map = {}
    for split in splits:
        for m, m_name in discrimination_metrics_mapping.items():
            val_col = f"{m}_{split}"
            ci_col = f"CI_{m}_{split}"
            data[val_col] = data.apply(lambda x: f"{np.round(x[val_col], 2)}\n{round_ci(x[ci_col])}", axis=1)
            value_cols.extend([val_col, ci_col])
            metric_map[val_col] = m_name
            metric_map[ci_col] = f"{m_name}"
    columns = id_vars + value_cols

    # --- melt ---
    df_long = data[columns].melt(
        id_vars=id_vars,
        value_vars=value_cols,
        var_name="Metric_Source",
        value_name="Value",
    )

    # --- extract Set ---
    pattern = "(" + "|".join(splits) + ")$"
    df_long["Set"] = df_long["Metric_Source"].str.replace("^CI_", "", regex=True).str.extract(pattern)[0]
    df_long["Set"] = df_long["Set"].map(splits).fillna(df_long["Set"])

    # --- map metric names ---
    df_long["Metric"] = df_long["Metric_Source"].map(metric_map)
    df_long["est_ci"] = df_long["Metric_Source"].apply(lambda x: "Estimate" if "CI" not in x else "95% CI")

    # --- ordering ---
    metric_order = [f"{name}" for name in ["Harrell", "Uno", "Antolini"]] + \
                   [f"AUC ({t})" for t in auc_times]

    df_long["Metric"] = pd.Categorical(df_long["Metric"], categories=metric_order, ordered=True)
    df_long["Set"] = pd.Categorical(
        df_long["Set"],
        categories=list(splits.values()),
        ordered=True,
    )
    # --- pivot ---
    table = df_long.pivot_table(
        index=["Metric"],
        columns=["Set", "Model"],
        values="Value",
        aggfunc="first",
        observed=False,
    )
    # Change index order
    cols = table.index.tolist()
    multi_cols = [split_metric_name(c) for c in cols]
    table.index = pd.MultiIndex.from_tuples(multi_cols, names=["Metric Family", "Metric"])

    # Change columns order
    set_order = ['Internal Validation', 'Swiss Test', 'Vienna Test', 'Test sets combined (treated)']
    model_order = ['RSF', 'NAC']
    table = table.reindex(
        columns=sorted(
            table.columns,
            key=lambda x: (set_order.index(x[0]), model_order.index(x[1]))
        )
    )

    table.to_excel(join(OUT_FOLDER, "discrimination_table.xlsx"))

def calibration_table(res):
    data = res.copy()
    data = data[data["Model"] == "RSF"]
    # --- metrics ---
    calibration_metrics_mapping = {}
    calibration_metrics = {
        "ici": "ICI",
        "mCalib": "Mean Calibration"
    }
    times = ["1y", "2y", "3y"]
    base_keys = list(calibration_metrics.keys())  # snapshot before loop
    for t in times:
        for metric in base_keys:
            calibration_metrics_mapping[f"{metric}_{t}"] = f"{calibration_metrics[metric]} ({t})"

    # --- columns to extract and map ---
    id_vars = ["Model"]
    value_cols = []
    metric_map = {}
    for split in splits:
        for m, m_name in calibration_metrics_mapping.items():
            val_col = f"{m}_{split}"
            ci_col = f"CI_{m}_{split}"
            data[ci_col] = data.apply(lambda x: f"{np.round(x[val_col], 2)}\n{round_ci(x[ci_col])}", axis=1)
            value_cols.extend([ci_col])
            metric_map[val_col] = m_name
            metric_map[ci_col] = f"{m_name}"
    columns = id_vars + value_cols

    # --- melt ---
    df_long = data[columns].melt(
        id_vars=id_vars,
        value_vars=value_cols,
        var_name="Metric_Source",
        value_name="Value",
    )

    # --- extract Set ---
    pattern = "(" + "|".join(splits) + ")$"
    df_long["Set"] = df_long["Metric_Source"].str.replace("^CI_", "", regex=True).str.extract(pattern)[0]
    df_long["Set"] = df_long["Set"].map(splits).fillna(df_long["Set"])

    # --- map metric names ---
    df_long["Metric"] = df_long["Metric_Source"].map(metric_map)

    # --- ordering ---
    metric_order = [f"Mean Calibration ({t})" for t in times] + \
                   [f"ICI ({t})" for t in times]

    df_long["Metric"] = pd.Categorical(df_long["Metric"], categories=metric_order, ordered=True)
    df_long["Set"] = pd.Categorical(
        df_long["Set"],
        categories=list(splits.values()),
        ordered=True,
    )
    # --- pivot ---
    table = df_long.pivot_table(
        index=["Metric"],
        columns=["Set"],
        values="Value",
        aggfunc="first",
        observed=False,
    )
    # current columns
    cols = table.index.tolist()

    # convert to list of tuples (level0, level1)
    multi_cols = [split_metric_name(c) for c in cols]

    # set MultiIndex
    table.index = pd.MultiIndex.from_tuples(multi_cols, names=["Metric", "Time"])
    table.to_excel(join(OUT_FOLDER, "calibration_table.xlsx"))


def make_tables():
    # Table 1
    df_complete = table1()
    missing_values_count(df_complete)
    # Combine results (RSF model and NAC)
    folders = ["rsf", "nac"]
    results = []
    for folder in folders:
        results.append(pd.read_excel(join(base_path, folder, "results_table.xlsx")))
    df_res = pd.concat(results)
    df_res.loc[df_res["Model"]=="CoxPH", "Model"] = "NAC"
    # Make tables
    discrimination_table(df_res)
    calibration_table(df_res)


if __name__ == "__main__":
    make_tables()