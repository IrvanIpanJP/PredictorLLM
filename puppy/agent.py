import os
import shutil
import pickle
import logging
from datetime import date
from typing import Dict, Union, Any, List

from abc import ABC, abstractmethod

from .run_type import RunMode
from .memorydb import BrainDB
from .portfolio import Portfolio
from .chat import get_chat_end_points
from .environment import market_info_type
from .reflection import trading_reflection

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler = logging.FileHandler("run.log", "a")
file_handler.setFormatter(logging_formatter)
logger.addHandler(file_handler)


class Agent(ABC):
    """
    Abstract base class for agents that can interact with a MarketEnvironment.
    """

    @abstractmethod
    def from_config(self, config: Dict[str, Any]) -> "Agent":
        """
        Create an Agent from a provided configuration dictionary.
        """
        pass

    @abstractmethod
    def train_step(self) -> None:
        """
        Execute one training step for the agent.
        """
        pass


class LLMAgent(Agent):
    """
    LLMAgent handles short-term, mid-term, long-term, and reflection memories for
    a specific symbol. It queries relevant memory, invokes a reflection LLM,
    logs the reflection, updates an internal Portfolio, and can store/reload its state.
    """

    def __init__(
        self,
        agent_name: str,
        trading_symbol: str,
        character_string: str,
        brain_db: BrainDB,
        top_k: int = 1,
        chat_end_point_name: str = "openai",
        chat_end_point_config: Union[Dict[str, Any], None] = None,
        look_back_window_size: int = 7,
    ):
        """
        Initialize the LLMAgent.

        Args:
            agent_name (str): Name of the agent.
            trading_symbol (str): Symbol the agent trades (stock, crypto, etc.).
            character_string (str): A query or "character" used to retrieve relevant memory.
            brain_db (BrainDB): Aggregated memory storage for short, mid, long, reflection layers.
            top_k (int, optional): Number of memory records to retrieve. Defaults to 1.
            chat_end_point_name (str, optional): Type of chat endpoint ("openai" or "together"). Defaults to "openai".
            chat_end_point_config (Dict[str, Any], optional): Configuration for the chosen chat endpoint.
            look_back_window_size (int, optional): Window used to compute feedback for memory importance. Defaults to 7.
        """
        if chat_end_point_config is None:
            chat_end_point_config = {"model_name": "gpt-4", "temperature": 0.7}

        self.counter = 1
        self.top_k = top_k
        self.agent_name = agent_name
        self.trading_symbol = trading_symbol
        self.character_string = character_string
        self.chat_end_point_name = chat_end_point_name
        self.chat_end_point_config = chat_end_point_config
        self.look_back_window_size = look_back_window_size

        # Brain DB
        self.brain = brain_db

        # Portfolio
        self.portfolio = Portfolio(
            symbol=self.trading_symbol,
            lookback_window_size=self.look_back_window_size
        )

        # Chat endpoints
        self.chat_end_point = get_chat_end_points(
            end_point_type=chat_end_point_name,
            chat_config=chat_end_point_config
        )
        self.guardrail_endpoint = self.chat_end_point.guardrail_endpoint()

        # Reflection records
        self.reflection_result_series_dict: Dict[date, Dict[str, Any]] = {}
        self.access_counter: Dict[str, int] = {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LLMAgent":
        """
        Construct an LLMAgent from a configuration dictionary.

        Returns:
            LLMAgent: The agent instance.
        """
        return cls(
            agent_name=config["general"]["agent_name"],
            trading_symbol=config["general"]["trading_symbol"],
            character_string=config["general"]["character_string"],
            brain_db=BrainDB.from_config(config=config),
            top_k=config["general"].get("top_k", 5),
            chat_end_point_name=config["chat"]["endpoint"],
            chat_end_point_config=config["chat"],
            look_back_window_size=config["general"]["look_back_window_size"],
        )

    def _handle_filings(self, cur_date: date, filing_q: str, filing_k: str) -> None:
        """
        Add fundamental text data to memory (mid or long) if present.

        Args:
            cur_date (date): Current date.
            filing_q (str): Q-like filing or fundamental data.
            filing_k (str): K-like filing or fundamental data.
        """
        if filing_q:
            self.brain.add_memory_mid(self.trading_symbol, cur_date, filing_q)
        if filing_k:
            self.brain.add_memory_long(self.trading_symbol, cur_date, filing_k)

    def _handle_news(self, cur_date: date, news: List[str]) -> None:
        """
        Add short-term news updates to memory.

        Args:
            cur_date (date): Current date.
            news (List[str]): News articles or headlines.
        """
        if news:
            self.brain.add_memory_short(self.trading_symbol, cur_date, news)

    def _query_info_for_reflection(self, run_mode: RunMode):
        """
        Query short, mid, long, and reflection memory from BrainDB
        for the relevant top_k items.
        """
        logger.info(f"Querying memory for symbol: {self.trading_symbol}")

        # Short
        short_queried, short_memory_id = self.brain.query_short(
            query_text=self.character_string,
            top_k=self.top_k,
            symbol=self.trading_symbol
        )
        for idx, memory_text in zip(short_memory_id, short_queried):
            logger.info(f"ShortTerm - ID: {idx}, Memory: {memory_text}")

        # Mid
        mid_queried, mid_memory_id = self.brain.query_mid(
            query_text=self.character_string,
            top_k=self.top_k,
            symbol=self.trading_symbol
        )
        for idx, memory_text in zip(mid_memory_id, mid_queried):
            logger.info(f"MidTerm - ID: {idx}, Memory: {memory_text}")

        # Long
        long_queried, long_memory_id = self.brain.query_long(
            query_text=self.character_string,
            top_k=self.top_k,
            symbol=self.trading_symbol
        )
        for idx, memory_text in zip(long_memory_id, long_queried):
            logger.info(f"LongTerm - ID: {idx}, Memory: {memory_text}")

        # Reflection
        reflection_queried, reflection_memory_id = self.brain.query_reflection(
            query_text=self.character_string,
            top_k=self.top_k,
            symbol=self.trading_symbol
        )
        for idx, memory_text in zip(reflection_memory_id, reflection_queried):
            logger.info(f"ReflectionTerm - ID: {idx}, Memory: {memory_text}")

        if run_mode == RunMode.Test:
            # For test mode, also retrieve momentum info
            momentum_ret = self.portfolio.get_moment(moment_window=2)
            momentum_value = momentum_ret["moment"] if momentum_ret else None
            return (
                short_queried, short_memory_id,
                mid_queried, mid_memory_id,
                long_queried, long_memory_id,
                reflection_queried, reflection_memory_id,
                momentum_value
            )
        else:
            return (
                short_queried, short_memory_id,
                mid_queried, mid_memory_id,
                long_queried, long_memory_id,
                reflection_queried, reflection_memory_id
            )

    def _reflect(
        self,
        cur_date: date,
        run_mode: RunMode,
        future_diff: Union[float, None] = None,
    ) -> None:
        """
        Invoke the reflection LLM for the given date and run mode. 
        Stores the reflection result in reflection_result_series_dict.
        """
        if run_mode == RunMode.Train:
            (
                short_queried, short_memory_id,
                mid_queried, mid_memory_id,
                long_queried, long_memory_id,
                reflection_queried, reflection_memory_id
            ) = self._query_info_for_reflection(run_mode=run_mode)

            reflection_result = trading_reflection(
                cur_date=cur_date,
                symbol=self.trading_symbol,
                run_mode=run_mode,
                endpoint_func=self.guardrail_endpoint,
                short_memory=short_queried,
                short_memory_id=short_memory_id,
                mid_memory=mid_queried,
                mid_memory_id=mid_memory_id,
                long_memory=long_queried,
                long_memory_id=long_memory_id,
                reflection_memory=reflection_queried,
                reflection_memory_id=reflection_memory_id,
                future_record=future_diff if future_diff is not None else 0.0
            )

        else:  # run_mode == RunMode.Test
            (
                short_queried, short_memory_id,
                mid_queried, mid_memory_id,
                long_queried, long_memory_id,
                reflection_queried, reflection_memory_id,
                momentum_value
            ) = self._query_info_for_reflection(run_mode=run_mode)

            reflection_result = trading_reflection(
                cur_date=cur_date,
                symbol=self.trading_symbol,
                run_mode=run_mode,
                endpoint_func=self.guardrail_endpoint,
                short_memory=short_queried,
                short_memory_id=short_memory_id,
                mid_memory=mid_queried,
                mid_memory_id=mid_memory_id,
                long_memory=long_queried,
                long_memory_id=long_memory_id,
                reflection_memory=reflection_queried,
                reflection_memory_id=reflection_memory_id,
                momentum=momentum_value,
            )

        if reflection_result and ("summary_reason" in reflection_result):
            self.brain.add_memory_reflection(
                symbol=self.trading_symbol,
                date=cur_date,
                text=reflection_result["summary_reason"]
            )
        else:
            logger.info("No reflection result or it did not converge properly.")

        self.reflection_result_series_dict[cur_date] = reflection_result

        # Log reflection
        if run_mode == RunMode.Train:
            logger.info(
                f"{self.trading_symbol} - {cur_date} (Train)\n"
                f"Reflection Summary: {reflection_result.get('summary_reason')}\n"
            )
        else:  # Test
            if reflection_result:
                logger.info(
                    f"{self.trading_symbol} - {cur_date} (Test)\n"
                    f"Decision: {reflection_result.get('investment_decision')}\n"
                    f"Reason: {reflection_result.get('summary_reason')}\n"
                )

    def _construct_train_actions(self, future_diff: float) -> Dict[str, int]:
        """
        For training: define an action based on future price difference.

        Args:
            future_diff (float): Next-step price difference.

        Returns:
            Dict[str, int]: e.g. {"direction": +1, "quantity": 1} if price is expected to rise.
        """
        direction = 1 if future_diff > 0 else -1
        return {"direction": direction, "quantity": 1}

    def _portfolio_step(self, action: Dict[str, int]) -> None:
        """
        Record the action in the portfolio, then update portfolio series.
        """
        self.portfolio.record_action(action=action)
        self.portfolio.update_portfolio_series()

    def _update_access_counter(self) -> None:
        """
        Update memory importance scores based on portfolio feedback signals.
        """
        feedback = self.portfolio.get_feedback_response()
        if not feedback:
            return

        if feedback["feedback"] != 0:
            cur_date = feedback["date"]
            reflection_record = self.reflection_result_series_dict.get(cur_date, {})
            if reflection_record:
                self._update_memory_access_by_layer(feedback, reflection_record)

    def _update_memory_access_by_layer(
        self,
        feedback: Dict[str, Union[int, date]],
        reflection_record: Dict[str, Any],
    ) -> None:
        """
        Update memory access counters across different memory layers
        (short, mid, long, reflection) if IDs are found in the reflection record.
        """
        def _update_sub(layer_key: str):
            if layer_key in reflection_record:
                layer_items = reflection_record[layer_key]
                if isinstance(layer_items, list):
                    memory_ids = [item["memory_index"] for item in layer_items]
                    self.brain.update_access_count_with_feed_back(
                        symbol=self.trading_symbol,
                        ids=memory_ids,
                        feedback=feedback["feedback"],  # type: ignore
                    )

        _update_sub("short_memory_index")
        _update_sub("middle_memory_index")
        _update_sub("long_memory_index")
        _update_sub("reflection_memory_index")

    @staticmethod
    def _process_test_action(reflection_result: Dict[str, Any]) -> Dict[str, int]:
        """
        Convert reflection test result (buy/sell/hold) to a trade action.

        Args:
            reflection_result (Dict[str, Any]): Output from reflection with 'investment_decision'.

        Returns:
            Dict[str, int]: The direction of the action, e.g., {"direction": 1} for buy.
        """
        decision = reflection_result.get("investment_decision")
        if decision == "buy":
            return {"direction": 1}
        elif decision == "hold":
            return {"direction": 0}
        elif decision == "sell":
            return {"direction": -1}
        else:
            # fallback
            return {"direction": 0}

    def step(
        self,
        market_info: market_info_type,
        run_mode: RunMode,
    ) -> None:
        """
        Perform a single step of environment + reflection for the agent.

        Args:
            market_info (market_info_type): Contains:
                (cur_date, cur_price, filing_k, filing_q, news, future_diff, done)
            run_mode (RunMode): Whether training or testing.

        Raises:
            ValueError: If run_mode is not Train or Test.
        """
        if run_mode not in [RunMode.Train, RunMode.Test]:
            raise ValueError("run_mode should be either RunMode.Train or RunMode.Test")

        (
            cur_date,
            cur_price,
            filing_k,
            filing_q,
            news,
            future_diff,
            done
        ) = market_info

        if done:
            return

        # 1. Handle fundamental filings (mid/long)
        self._handle_filings(cur_date, filing_q, filing_k)

        # 2. Handle news (short memory)
        self._handle_news(cur_date, news)

        # 3. Update portfolio with market price
        self.portfolio.update_market_info(new_market_price_info=cur_price, cur_date=cur_date)

        # 4. LLM reflection
        self._reflect(cur_date=cur_date, run_mode=run_mode, future_diff=future_diff)

        # 5. Decide action
        if run_mode == RunMode.Train:
            action = self._construct_train_actions(future_diff=future_diff)  # type: ignore
        else:
            reflection_record = self.reflection_result_series_dict.get(cur_date, {})
            action = self._process_test_action(reflection_record)

        # 6. Update portfolio with the chosen action
        self._portfolio_step(action)

        # 7. Update memory importance
        self._update_access_counter()

        # 8. Step memory (decay, cleanup, jump)
        self.brain.step()

    def train_step(self) -> None:
        """
        Satisfy the abstract method. This could be a custom method for
        more advanced logic if needed. Currently unused.
        """
        pass

    def save_checkpoint(self, path: str, force: bool = False) -> None:
        """
        Save the agent state to disk, including the BrainDB.

        Args:
            path (str): Directory where agent state will be saved.
            force (bool): Whether to overwrite existing directory.
        """
        path = os.path.join(path, self.agent_name)
        if os.path.exists(path):
            if force:
                shutil.rmtree(path)
            else:
                raise FileExistsError(f"Path {path} already exists.")
        os.mkdir(path)

        # Brain DB
        os.mkdir(os.path.join(path, "brain"))

        state_dict = {
            "agent_name": self.agent_name,
            "character_string": self.character_string,
            "top_k": self.top_k,
            "counter": self.counter,
            "trading_symbol": self.trading_symbol,
            "chat_end_point_name": self.chat_end_point_name,
            "chat_end_point_config": self.chat_end_point_config,
            "portfolio": self.portfolio,
            "chat_end_point": self.chat_end_point,
            "reflection_result_series_dict": self.reflection_result_series_dict,
            "access_counter": self.access_counter,
        }

        with open(os.path.join(path, "state_dict.pkl"), "wb") as f:
            pickle.dump(state_dict, f)

        self.brain.save_checkpoint(os.path.join(path, "brain"), force=force)

    @classmethod
    def load_checkpoint(cls, path: str) -> "LLMAgent":
        """
        Load an LLMAgent state from disk.

        Args:
            path (str): The path to the saved agent directory.

        Returns:
            LLMAgent: A restored LLMAgent instance.
        """
        with open(os.path.join(path, "state_dict.pkl"), "rb") as f:
            state_dict = pickle.load(f)

        brain = BrainDB.load_checkpoint(os.path.join(path, "brain"))

        loaded_agent = cls(
            agent_name=state_dict["agent_name"],
            trading_symbol=state_dict["trading_symbol"],
            character_string=state_dict["character_string"],
            brain_db=brain,
            top_k=state_dict["top_k"],
            chat_end_point_name=state_dict["chat_end_point_name"],
            chat_end_point_config=state_dict["chat_end_point_config"],
        )
        # Restore the agent attributes
        loaded_agent.chat_end_point = state_dict["chat_end_point"]
        loaded_agent.portfolio = state_dict["portfolio"]
        loaded_agent.reflection_result_series_dict = state_dict["reflection_result_series_dict"]
        loaded_agent.access_counter = state_dict["access_counter"]
        loaded_agent.counter = state_dict["counter"]

        return loaded_agent
