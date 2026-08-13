import pandas as pd

from scripts.collect_results import agg_mean_std_ci


def test_agg_mean_std_ci_skips_structurally_present_but_empty_metrics():
    df = pd.DataFrame(
        {
            "boundary_mode": ["closed", "closed", "half_open", "half_open"],
            "ms_lut_numpy": [None, None, None, None],
            "in_mae": [1.0e-5, 2.0e-5, 3.0e-5, 4.0e-5],
        }
    )

    out = agg_mean_std_ci(df, ["boundary_mode"], ["ms_lut_numpy"])

    assert list(out.columns) == ["boundary_mode", "n__count"]
    assert sorted(out["n__count"].tolist()) == [2, 2]


def test_agg_mean_std_ci_keeps_numeric_metrics_when_other_candidates_are_empty():
    df = pd.DataFrame(
        {
            "boundary_mode": ["closed", "closed", "half_open", "half_open"],
            "empty_metric": [None, None, None, None],
            "in_mae": [1.0e-5, 2.0e-5, 3.0e-5, 4.0e-5],
        }
    )

    out = agg_mean_std_ci(
        df,
        ["boundary_mode"],
        ["empty_metric", "in_mae"],
    )

    assert "empty_metric__mean" not in out.columns
    assert "in_mae__mean" in out.columns
    assert "in_mae__std" in out.columns
    assert "in_mae__ci95" in out.columns
