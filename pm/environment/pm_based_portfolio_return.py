import numpy as np
from typing import List, Any
from sklearn.preprocessing import StandardScaler
import random
import gymnasium as gym

from pm.registry import ENVIRONMENT


@ENVIRONMENT.register_module()
class EnvironmentRET(gym.Env):
    def __init__(self,
                 mode: str = "train",
                 dataset: Any = None,
                 if_norm: bool = True,
                 if_norm_temporal: bool = True,
                 scaler: List[StandardScaler] = None,
                 days: int = 10,
                 start_date: str = None,
                 end_date: str = None,
                 initial_amount: int = 1e3,
                 transaction_cost_pct: float = 1e-3
                 ):
        super().__init__()

        self.mode = mode
        self.dataset = dataset
        self.if_norm = if_norm
        self.if_norm_temporal = if_norm_temporal
        self.scaler = scaler
        self.days = days
        self.start_date = start_date
        self.end_date = end_date
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct

        if end_date is not None:
            assert end_date > start_date, "start date {}, end date {}, end date should be greater than start date".format(start_date, end_date)

        self.stocks = self.dataset.stocks
        self.stocks2id = self.dataset.stocks2id
        self.id2stocks = self.dataset.id2stocks
        self.aux_stocks = self.dataset.aux_stocks

        self.features_name = self.dataset.features_name
        self.prices_name = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
        self.temporals_name = self.dataset.temporals_name
        self.labels_name = self.dataset.labels_name
        self.stocks_df = []

        prices = []

        if if_norm:
            print("normalize datasets")
            self.scaler = []

            for i, df in enumerate(self.dataset.stocks_df):
                df = df.loc[start_date:end_date] if end_date else df.loc[start_date:]
                df[self.prices_name] = df[[name.lower() for name in self.prices_name]]
                prices.append(df[self.prices_name].values)

                scaler = StandardScaler()
                if self.if_norm_temporal:
                    df[self.features_name + self.temporals_name] = scaler.fit_transform(df[self.features_name + self.temporals_name])
                else:
                    df[self.features_name] = scaler.fit_transform(df[self.features_name])

                self.scaler.append(scaler)
                self.stocks_df.append(df)
        else:
            print("no normalize datasets")

        self.features = np.stack([df[self.features_name + self.temporals_name].values for df in self.stocks_df])
        self.prices = np.stack(prices)
        self.labels = np.stack([df[self.labels_name].values for df in self.stocks_df])

        print(f"features shape {self.features.shape}, prices shape {self.prices.shape}, labels shape {self.labels.shape}, num days {self.features.shape[1]}")

        self.num_days = self.features.shape[1]
        self.day = random.randint(0, 3 * (self.num_days // 4)) if self.mode == "train" else 0

    def get_current_date(self):
        return self.stocks_df[0].index[self.day]

    def reset(self, seed=None, options=None):
        if self.mode == "train":
            self.day = random.randint(0, 3 * (self.num_days // 4))
        else:
            self.day = 0

        state = self.features[:, self.day: self.day + self.days, :]
        self.state = state

        return state, {}

    def step(self, action: np.array = None):
        weights = action.flatten()
        labels_ret = self.labels[:, self.day + self.days - 1, 0].flatten()

        portfolio_ret = np.sum(weights[1:] * labels_ret)
        reward = portfolio_ret

        self.day += 1
        terminated = self.day + self.days >= self.num_days
        truncated = False

        next_state = self.features[:, self.day: self.day + self.days, :]
        self.state = next_state

        info = {
            "state": self.state,
            "action": action,
            "portfolio_ret": portfolio_ret,
        }

        return next_state, reward, terminated, truncated, info
