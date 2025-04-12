import numpy as np
import pandas as pd
import pytest
from app.risk.risk_metrics import RiskMetrics


# ---- Fixtures ----
@pytest.fixture
def sample_returns():
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=300, freq="B")
    data = {
        "Asset_A": np.random.normal(0, 0.01, size=300),
        "Asset_B": np.random.normal(0, 0.015, size=300),
    }
    return pd.DataFrame(data, index=dates)


# ---- Tests ----


def test_initialization(sample_returns):
    metrics = RiskMetrics(df=sample_returns)
    assert isinstance(metrics.df, pd.DataFrame)


def test_compute_var_structure(sample_returns):
    metrics = RiskMetrics(df=sample_returns, lookback_period=50)
    var_df = metrics.compute_var()
    assert "VaR_Asset_A" in var_df.columns
    assert "VaR_Asset_B" in var_df.columns
    assert var_df.shape[0] < sample_returns.shape[0]  # Because of lookback NaNs


def test_compute_cvar_structure(sample_returns):
    metrics = RiskMetrics(df=sample_returns, lookback_period=50)
    cvar_df = metrics.compute_cvar()
    assert "CVaR_Asset_A" in cvar_df.columns
    assert "CVaR_Asset_B" in cvar_df.columns
    assert cvar_df.shape[0] == sample_returns.shape[0]


def test_backtest_var_output(sample_returns):
    metrics = RiskMetrics(df=sample_returns, lookback_period=50)
    summary = metrics.backtest_var()
    assert isinstance(summary, dict)
    assert "Asset_A" in summary
    assert "Number of breaches" in summary["Asset_A"]
    assert metrics.breaches is not None
    assert len(metrics.breaches["Asset_A"]) > 0
