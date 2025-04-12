import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class RiskMetrics:
    def __init__(
        self,
        df: pd.DataFrame = None,
        confidence_level: float = 0.95,
        lookback_period: int = 252,
        method: str = "historical",
    ):
        self.df = df
        self.confidence_level = confidence_level
        self.lookback_period = lookback_period
        self.method = method

        if self.df is None or self.df.empty:
            raise (ValueError("No dataframe or empty dataframe passed."))

    def compute_var(self):
        if self.method == "historical":
            var_df = pd.DataFrame(index=self.df.index)

            for asset in self.df.columns:
                var_df[f"VaR_{asset}"] = (
                    self.df[asset]
                    .rolling(window=self.lookback_period)
                    .quantile(1.0 - self.confidence_level)
                )

            self.var_df = pd.concat([self.df, var_df], axis=1).iloc[
                self.lookback_period + 1 :
            ]  # remove the first loockback periods nan (true for all assets)

            return self.var_df

    @staticmethod
    def _historical_cvar(arr, alpha):

        cvar_threshold = np.quantile(arr, 1 - alpha)
        tail_losses = arr[arr <= cvar_threshold]

        if len(tail_losses) == 0:
            return np.nan

        return tail_losses.mean()

    def compute_cvar(self):
        if self.method == "historical":
            cvar_df = pd.DataFrame(index=self.df.index)

            for asset in self.df.columns:
                cvar_df[f"CVaR_{asset}"] = (
                    self.df[asset]
                    .rolling(window=self.lookback_period)
                    .apply(
                        lambda x: self._historical_cvar(x, self.confidence_level),
                        raw=True,
                    )
                )

            return cvar_df

    def compute_all(self):
        var_df = self.compute_var()
        cvar_df = self.compute_cvar()

        return pd.concat([self.df, var_df, cvar_df], axis=1)

    def backtest_var(self):

        var_df = self.compute_var()

        breaches = {}
        breaches_summary = {}

        assets = self.df.columns
        for asset in assets:

            breach_mask = var_df[asset] < var_df[f"VaR_{asset}"]
            breaches[asset] = breach_mask

            number_of_breaches = breach_mask.sum()
            number_of_observations = len(breach_mask)
            breach_rate = number_of_breaches / number_of_observations

            expected_breaches = (1 - self.confidence_level) * number_of_observations

            breaches_summary[asset] = {
                "Number of breaches": number_of_breaches,
                "Expected breaches": expected_breaches,
                "Breach rate": breach_rate,
                "Number of observations": number_of_observations,
            }

        self.breaches = breaches

        return breaches_summary

    def plot_var_breaches(self, asset: str):

        if not hasattr(self, "breaches"):
            self.backtest_var()

        breaches = self.breaches[asset]

        df = self.var_df

        plt.figure(figsize=(12, 5))
        plt.plot(df.index, df[asset], label="Returns")
        plt.plot(
            df.index,
            df[f"VaR_{asset}"],
            label=f"{int(self.confidence_level*100)}% VaR",
            linestyle="--",
        )
        plt.scatter(
            df.index[breaches],
            df[asset][breaches],
            color="red",
            label="Breaches",
            zorder=5,
        )

        plt.title(f"VaR Breaches for {asset}")
        plt.legend()
        plt.tight_layout()

        plt.show()
