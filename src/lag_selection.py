import ast
import pandas as pd


def _as_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return [x]
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, tuple):
                return [parsed]
        except Exception:
            return []
    return []


def _longest_consecutive_run(lags):
    lags = sorted(set(int(x) for x in lags))
    if not lags:
        return []

    best = []
    current = [lags[0]]

    for lag in lags[1:]:
        if lag == current[-1] + 1:
            current.append(lag)
        else:
            if len(current) > len(best):
                best = current
            current = [lag]

    if len(current) > len(best):
        best = current

    return best


def lag_stability_summary(
    sensitivity: pd.DataFrame,
    *,
    min_consecutive: int = 2,
    min_total: int = 3,
) -> pd.DataFrame:
    rows = []

    for _, row in sensitivity.iterrows():
        if "error" in row and pd.notna(row["error"]):
            continue

        backend = row.get("backend")
        run = row.get("run")
        lag = int(row.get("lag"))

        for pair in _as_list(row.get("pairs")):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue

            cause, trigger = pair

            rows.append({
                "backend": backend,
                "run": run,
                "cause": cause,
                "trigger": trigger,
                "lag": lag,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "backend", "run", "cause", "trigger",
            "lags", "n_lags", "longest_consecutive_lags",
            "n_consecutive", "representative_lag", "lag_stable",
        ])

    pair_rows = pd.DataFrame(rows)

    summaries = []
    for keys, group in pair_rows.groupby(["backend", "run", "cause", "trigger"]):
        lags = sorted(group["lag"].unique().tolist())
        longest = _longest_consecutive_run(lags)

        summaries.append({
            "backend": keys[0],
            "run": keys[1],
            "cause": keys[2],
            "trigger": keys[3],
            "lags": lags,
            "n_lags": len(lags),
            "longest_consecutive_lags": longest,
            "n_consecutive": len(longest),
            "representative_lag": min(longest) if longest else min(lags),
            "lag_stable": (len(longest) >= min_consecutive) or (len(lags) >= min_total),
        })

    out = pd.DataFrame(summaries)

    return out.sort_values(
        ["lag_stable", "n_consecutive", "n_lags", "representative_lag"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def select_stable_lag_summary(stability: pd.DataFrame) -> pd.DataFrame:
    if stability is None or stability.empty:
        return pd.DataFrame()

    stable = stability[stability["lag_stable"]].copy()
    if stable.empty:
        return stable

    return stable.sort_values(
        ["n_consecutive", "n_lags", "representative_lag"],
        ascending=[False, False, True],
    ).reset_index(drop=True)