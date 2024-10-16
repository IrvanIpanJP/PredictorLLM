import os
import shutil
import pickle
from datetime import date
from typing import List, Dict, Tuple, Union, Any
from pydantic import BaseModel, ValidationError

# Type alias for one environment step
market_info_type = Tuple[
    date,       # current date
    float,      # current asset price
    str,        # filing_k
    str,        # filing_q
    List[str],  # news
    float,      # future difference
    bool,       # done flag
]
terminated_market_info_type = Tuple[None, None, None, None, None, None, bool]


class OneDateRecord(BaseModel):
    """
    Validates the structure for a single date in the environment data.
    """
    price: Dict[str, float]
    filing_k: Dict[str, str]
    filing_q: Dict[str, str]
    news: Dict[str, List[str]]


class MarketEnvironment:
    """
    Manages historical (or synthetic) data for a single symbol. Each date has:
        - price
        - filing_k
        - filing_q
        - news
    The environment steps through the dates, returning data for each step.
    """

    def __init__(
        self,
        env_data_pkl: Dict[date, Dict[str, Any]],
        start_date: date,
        end_date: date,
        symbol: str,
    ) -> None:
        if not env_data_pkl:
            raise ValueError("env_data_pkl cannot be empty.")

        first_date = list(env_data_pkl.keys())[0]
        if not isinstance(first_date, date):
            raise TypeError("env_data_pkl keys must be of type datetime.date")

        try:
            OneDateRecord.model_validate(env_data_pkl[first_date])
        except ValidationError as e:
            raise ValueError(f"Failed validation for initial date record: {e}")

        self.date_series_full = sorted(env_data_pkl.keys())
        if (start_date not in self.date_series_full) or (end_date not in self.date_series_full):
            raise ValueError("start_date and end_date must exist in env_data_pkl keys.")

        # Keep only the slice of dates
        self.date_series = [d for d in self.date_series_full if (start_date <= d <= end_date)]
        self.start_date = start_date
        self.end_date = end_date
        self.cur_date = None
        self.env_data = env_data_pkl
        self.symbol = symbol

        self.simulation_length = len(self.date_series)
        self.date_series_keep = self.date_series.copy()

    def reset(self) -> None:
        """
        Reset the environment to the start_date -> end_date date range.
        """
        self.date_series = [
            d for d in self.date_series_keep
            if (self.start_date <= d <= self.end_date)
        ]
        self.cur_date = None

    def step(self) -> Union[market_info_type, terminated_market_info_type]:
        """
        Return one timestep of data:
          - (cur_date, cur_price, filing_k, filing_q, news, future_record, done_flag)
        or (None, None, None, None, None, None, True) if we reach the end.

        future_record is the price difference (symbol) from current day to next day.
        """
        if not self.date_series:
            return None, None, None, None, None, None, True

        # pop the first date
        self.cur_date = self.date_series.pop(0)

        if not self.date_series:
            return None, None, None, None, None, None, True

        cur_date = self.cur_date
        next_date = self.date_series[0]
        cur_entry = self.env_data[cur_date]
        next_entry = self.env_data[next_date]

        # symbol price
        cur_price = cur_entry["price"].get(self.symbol, 0.0)
        next_price = next_entry["price"].get(self.symbol, 0.0)
        filing_k = cur_entry["filing_k"].get(self.symbol, "")
        filing_q = cur_entry["filing_q"].get(self.symbol, "")
        news_list = cur_entry["news"].get(self.symbol, [])
        future_diff = next_price - cur_price

        return (
            cur_date,
            cur_price,
            filing_k,
            filing_q,
            news_list,
            future_diff,
            False,
        )

    def save_checkpoint(self, path: str, force: bool = False) -> None:
        """
        Save the entire MarketEnvironment to disk, so the simulation state can be resumed.
        """
        path = os.path.join(path, "env")
        if os.path.exists(path):
            if not force:
                raise FileExistsError(f"Path {path} already exists.")
            shutil.rmtree(path)
        os.mkdir(path)
        with open(os.path.join(path, "env.pkl"), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_checkpoint(cls, path: str) -> "MarketEnvironment":
        """
        Load the MarketEnvironment from disk.

        Args:
            path (str): Path to the folder containing "env.pkl".

        Returns:
            MarketEnvironment: The restored environment instance.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist.")
        with open(os.path.join(path, "env.pkl"), "rb") as f:
            env = pickle.load(f)
        env.simulation_length = len(env.date_series)
        return env
