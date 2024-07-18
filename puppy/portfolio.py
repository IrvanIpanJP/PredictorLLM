import polars as pl
import numpy as np
from datetime import date
from annotated_types import Gt
from typing import Dict, Annotated, Union
from pydantic import BaseModel


class PriceStructure(BaseModel):
    price: Annotated[float, Gt(0)]


class Portfolio:
    """
    Tracks an agent’s position in a single symbol (stock, crypto, etc.).
    'holding_shares' can represent integer shares or coins. 
    For crypto, if you want fractional positions, you could adapt the code 
    to store float directions or quantities.
    """

    def __init__(self, symbol: str, lookback_window_size: int = 7) -> None:
        self.cur_date = None
        self.symbol = symbol
        self.action_series = {}
        self.market_price = None
        self.day_count = 0
        self.date_series = []
        self.holding_shares = 0
        self.market_price_series = np.array([])
        self.portfolio_share_series = np.array([])
        self.lookback_window_size = lookback_window_size

    def update_market_info(self, new_market_price_info: float, cur_date: date) -> None:
        PriceStructure.model_validate(new_market_price_info)
        self.market_price = new_market_price_info
        self.cur_date = cur_date
        self.date_series.append(cur_date)
        self.day_count += 1
        self.market_price_series = np.append(self.market_price_series, new_market_price_info)

    def record_action(self, action: Dict[str, int]) -> None:
        """
        For each step, you can record an action: 
        action['direction'] = +1 (buy), -1 (sell), 0 (hold).
        If you want to handle fractional crypto trades, adapt accordingly.
        """
        self.holding_shares += action.get("direction", 0)
        self.action_series[self.cur_date] = action.get("direction", 0)

    def get_action_df(self) -> pl.DataFrame:
        temp_dict = {"date": [], "symbol": [], "direction": []}
        for d in self.action_series:
            temp_dict["date"].append(d)
            temp_dict["symbol"].append(self.symbol)
            temp_dict["direction"].append(self.action_series[d])
        return pl.DataFrame(temp_dict)

    def update_portfolio_series(self) -> None:
        self.portfolio_share_series = np.append(self.portfolio_share_series, self.holding_shares)

    def get_feedback_response(self) -> Union[Dict[str, Union[int, date]], None]:
        """
        Looks back 'lookback_window_size' steps. 
        If cumulative PnL in that window is >0 => feedback=+1, <0 => -1, else 0.
        This feedback is used to adjust memory importance (reinforcement).
        """
        if self.day_count <= self.lookback_window_size:
            return None

        price_changes = np.diff(self.market_price_series)
        # Typically we compare price changes with the # of shares held at each point
        if len(price_changes) != len(self.portfolio_share_series[:-1]):
            # One off to align indexes
            temp_pnl = np.cumsum(
                (
                    price_changes[:-1] * self.portfolio_share_series[:-1]
                )[-self.lookback_window_size :]
            )[-1]
        else:
            temp_pnl = np.cumsum(
                (price_changes * self.portfolio_share_series[:-1])[-self.lookback_window_size :]
            )[-1]

        if temp_pnl > 0:
            return {
                "feedback": 1,
                "date": self.date_series[-self.lookback_window_size],
            }
        elif temp_pnl < 0:
            return {
                "feedback": -1,
                "date": self.date_series[-self.lookback_window_size],
            }
        else:
            return {
                "feedback": 0,
                "date": self.date_series[-self.lookback_window_size],
            }

    def get_moment(self, moment_window: int = 3) -> Union[Dict[str, int], None]:
        """
        Return a simple sign of the last X days' price changes.
        If sum of last X price changes >0 => 1, <0 => -1, else 0
        """
        if self.day_count <= moment_window:
            return None

        recent_change = np.cumsum(np.diff(self.market_price_series)[-moment_window:])[-1]
        if recent_change > 0:
            return {"moment": 1, "date": self.date_series[-moment_window]}
        elif recent_change < 0:
            return {"moment": -1, "date": self.date_series[-moment_window]}
        else:
            return {"moment": 0, "date": self.date_series[-moment_window]}
