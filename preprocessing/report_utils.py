import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def rkf_summary(time, event, unit_months=False):
    """
    Reverse Kaplan–Meier: median [Q1, Q3] follow-up time
    event: 1 = event (MACE), 0 = censored
    """
    kmf = KaplanMeierFitter()
    kmf.fit(time, event_observed=1 - event)

    sf = kmf.survival_function_.reset_index()
    sf.columns = ["time", "survival"]

    def get_quantile(p):
        q = sf.loc[sf["survival"] <= 1 - p, "time"]
        return q.iloc[0] if len(q) > 0 else sf["time"].max()

    q1 = get_quantile(0.25)
    med = get_quantile(0.50)
    q3 = get_quantile(0.75)

    if unit_months:
        q1 /= 30.4375
        med /= 30.4375
        q3 /= 30.4375

    return f"{med:.1f} [{q1:.1f}, {q3:.1f}]"


def km_median(time, event, unit_months=False):
    """
    Kaplan–Meier median survival time
    Returns 'NA' if median not reached
    """
    kmf = KaplanMeierFitter()
    kmf.fit(time, event_observed=event)

    median = kmf.median_survival_time_
    if unit_months and median is not None:
        median /= 30.4375
    return f"{median:.1f}" if median is not None else "NA"


def km_survival_at_time(time, event, t):
    """
    KM survival probability at time t with 95% CI, formatted as a single string
    Example: "87.0% [82.0%, 91.0%]"
    """
    kmf = KaplanMeierFitter()
    kmf.fit(time, event_observed=event)

    sf = kmf.survival_function_
    ci = kmf.confidence_interval_

    # last observed time <= t
    idx = sf.index[sf.index <= t].max()

    est = sf.loc[idx].iloc[0] * 100
    ci_low = ci.loc[idx, ci.columns[0]] * 100
    ci_high = ci.loc[idx, ci.columns[1]] * 100

    return f"{est:.1f} [{ci_low:.1f}, {ci_high:.1f}]"


def survival_summary_by_group(
    df,
    stratify_by,
    time_col,
    event_col,
    times=np.array([1, 2, 3]) * 365,
    output_path=None,
):
    """
    Create a survival summary table stratified by a given feature.

    Parameters
    ----------
    df : pd.DataFrame
    stratify_by : str
        Column name to stratify on (e.g. 'Center')
    time_col : str
        Follow-up / time-to-event column
    event_col : str
        Event indicator (1=event, 0=censored)
    times : tuple
        Time points for survival probabilities
    output_path : str or None
        If provided, saves table to Excel
    """
    g = df.groupby(stratify_by)
    table = pd.DataFrame({
        "n": g[time_col].count(),
        "Events (%)": g[event_col].apply(lambda x: f"{x.sum()} ({x.sum()/x.count()*100:.1f})"),
        "Median follow-up (days)": g.apply(lambda d: rkf_summary(d[time_col], d[event_col])),
        "Median follow-up (months)": g.apply(lambda d: rkf_summary(d[time_col], d[event_col], unit_months=True)) ,
        "Median survival (days)": g.apply(lambda d: km_median(d[time_col], d[event_col])),
        "Median survival (months)": g.apply(lambda d: km_median(d[time_col], d[event_col], unit_months=True)),
    })
    # Survival at fixed times
    for t in times:
        table[f"{t}-day survival"] = g.apply(
            lambda d: km_survival_at_time(
                d[time_col], d[event_col], t
            )
        )
    table.sort_values(by="n", ascending=False, inplace=True)
    if output_path is not None:
        table.to_excel(output_path)
    return table

def plot_kaplan_meier(datasets, labels, out_path, time_col, event_col, max_t_year=4, ci_show=True):
    fig, ax = plt.subplots(figsize=(8, 6))
    curves = []
    colors = ["#2A6F97", "#9A3D3F", '#7FB77E', "#5FA8D3", "#EE6C4D"][:len(datasets)]   # professional non-default colors
    for df_, set_name, color in zip(datasets, labels, colors):
        curve = KaplanMeierFitter()
        curve.fit(
            durations=df_[time_col] / 365.25,
            event_observed=df_[event_col],
            label=set_name
        )
        curve.plot_survival_function(
            ax=ax,
            linewidth=2,
            ci_show=ci_show,
            color=color,
            loc=slice(0., max_t_year)
        )
        curves.append(curve)
    add_at_risk_counts(*curves, ax=ax, fontsize=14, rows_to_show=['At risk'])
    # Labels
    ax.set_xlabel(xlabel="Years", fontsize=15)
    ax.set_ylabel(ylabel="Survival Probability", fontsize=15)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="gray", alpha=0.2)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    for curve, label in zip(curves, labels):
        print(f"\n{label}")
        for year in range(1, max_t_year+1):
            if year <= max_t_year:
                surv = curve.survival_function_at_times(year).values[0]
                ci_df = curve.confidence_interval_
                ci_lower = np.interp(year, ci_df.index, ci_df.iloc[:, 0])
                ci_upper = np.interp(year, ci_df.index, ci_df.iloc[:, 1])
                print(f"  {year}-year survival: {surv:.3f} (95% CI: {ci_lower:.3f}–{ci_upper:.3f})")
