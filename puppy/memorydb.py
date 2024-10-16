import os
import faiss
import pickle
import logging
import shutil
import numpy as np
from datetime import date
from itertools import repeat
from sortedcontainers import SortedList
from typing import List, Union, Dict, Any, Tuple, Callable

from .embedding import OpenAILongerThanContextEmb
from .memory_functions import (
    ImportanceScoreInitialization,
    get_importance_score_initialization_func,
    R_ConstantInitialization,
    LinearCompoundScore,
    ExponentialDecay,
    LinearImportanceScoreChange,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler = logging.FileHandler("run.log", mode="a")
file_handler.setFormatter(logging_formatter)
logger.addHandler(file_handler)


class IdGeneratorFunc:
    """
    Simple callable class to generate unique integer IDs for memory records.
    """
    def __init__(self):
        self.current_id = 0

    def __call__(self) -> int:
        """
        Generate the next ID (auto-incremented).

        Returns:
            int: The next unique ID.
        """
        self.current_id += 1
        return self.current_id - 1


class MemoryDB:
    """
    A memory layer for storing text data alongside vector embeddings, importance, and recency.
    Provides methods for adding text, querying by embedding similarity, and applying
    decays or clean-ups over time.
    """

    def __init__(
        self,
        db_name: str,
        id_generator: Callable[[], int],
        jump_threshold_upper: float,
        jump_threshold_lower: float,
        embedding_function: OpenAILongerThanContextEmb,
        importance_score_initialization: ImportanceScoreInitialization,
        recency_score_initialization: R_ConstantInitialization,
        compound_score_calculation: LinearCompoundScore,
        importance_score_change_access_counter: LinearImportanceScoreChange,
        decay_function: ExponentialDecay,
        clean_up_threshold_dict: Dict[str, float],
    ) -> None:
        """
        Initialize a memory layer.

        Args:
            db_name (str): Name of this memory database (e.g., "agent_short").
            id_generator (Callable): A function that generates unique IDs.
            jump_threshold_upper (float): If importance >= this, the record may jump to a higher memory layer.
            jump_threshold_lower (float): If importance < this, the record may jump to a lower memory layer.
            embedding_function (OpenAILongerThanContextEmb): Embedding function for text.
            importance_score_initialization (ImportanceScoreInitialization): Initialization of importance score.
            recency_score_initialization (R_ConstantInitialization): Initialization of recency score.
            compound_score_calculation (LinearCompoundScore): Combines importance & recency.
            importance_score_change_access_counter (LinearImportanceScoreChange): Updates importance with feedback.
            decay_function (ExponentialDecay): Applies decays each step.
            clean_up_threshold_dict (Dict[str, float]): Thresholds for cleaning up old records.
        """
        self.db_name = db_name
        self.id_generator = id_generator
        self.jump_threshold_upper = jump_threshold_upper
        self.jump_threshold_lower = jump_threshold_lower

        self.embedding_dim = embedding_function.get_embedding_dimension()
        self.embedding_function = embedding_function

        self.importance_score_init_func = importance_score_initialization
        self.recency_score_init_func = recency_score_initialization
        self.compound_score_calc_func = compound_score_calculation
        self.decay_function = decay_function
        self.importance_score_change_access_counter = importance_score_change_access_counter
        self.clean_up_threshold_dict = clean_up_threshold_dict

        # Universe: symbol -> {"score_memory": SortedList, "index": faiss Index}
        self.universe: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        agent_name: str,
        memory_layer: str
    ) -> "MemoryDB":
        """
        Alternate constructor from a config dict. Typically for convenience.
        """
        return cls(
            db_name=f"{agent_name}_{memory_layer}",
            id_generator=IdGeneratorFunc(),
            jump_threshold_upper=config[memory_layer]["jump_threshold_upper"],
            jump_threshold_lower=config[memory_layer]["jump_threshold_lower"],
            embedding_function=OpenAILongerThanContextEmb(**config["embedding"]),
            importance_score_initialization=get_importance_score_initialization_func(
                type=config[memory_layer]["importance_score_initialization_type"],
                memory_layer=memory_layer
            ),
            recency_score_initialization=R_ConstantInitialization(),
            compound_score_calculation=LinearCompoundScore(),
            importance_score_change_access_counter=LinearImportanceScoreChange(),
            decay_function=ExponentialDecay(**config[memory_layer]["decay_params"]),
            clean_up_threshold_dict=config[memory_layer]["clean_up_threshold_dict"],
        )

    def add_new_symbol(self, symbol: str) -> None:
        """
        Register a new symbol in the memory. Creates an empty Faiss index and
        an empty SortedList for storing memory records.
        """
        index = faiss.IndexFlatIP(self.embedding_dim)
        index = faiss.IndexIDMap2(index)
        temp_record = {
            "score_memory": SortedList(
                key=lambda x: x["importance_recency_compound_score"]
            ),
            "index": index,
        }
        self.universe[symbol] = temp_record

    def add_memory(self, symbol: str, date_added: date, text: Union[List[str], str]) -> None:
        """
        Add text data to the memory for a given symbol.

        Args:
            symbol (str): Symbol to which this text belongs.
            date_added (date): Date when this text was added.
            text (List[str] or str): One or more text strings to store.
        """
        if symbol not in self.universe:
            self.add_new_symbol(symbol)

        if isinstance(text, str):
            text = [text]

        embeddings = self.embedding_function(text)
        faiss.normalize_L2(embeddings)
        ids = [self.id_generator() for _ in range(len(text))]

        # Initialize importance & recency
        importance_scores = [self.importance_score_init_func() for _ in text]
        recency_scores = [self.recency_score_init_func() for _ in text]

        compound_scores = [
            self.compound_score_calc_func.recency_and_importance_score(
                recency_score=r_s,
                importance_score=i_s
            )
            for i_s, r_s in zip(importance_scores, recency_scores)
        ]

        self.universe[symbol]["index"].add_with_ids(embeddings, np.array(ids))

        for i, t in enumerate(text):
            new_record = {
                "text": t,
                "id": ids[i],
                "importance_score": importance_scores[i],
                "recency_score": recency_scores[i],
                "delta": 0,
                "importance_recency_compound_score": compound_scores[i],
                "access_counter": 0,
                "date": date_added,
            }
            self.universe[symbol]["score_memory"].add(new_record)
            logger.info(f"Added memory: {new_record}")

    def query(
        self,
        query_text: str,
        top_k: int,
        symbol: str
    ) -> Tuple[List[str], List[int]]:
        """
        Retrieve up to top_k relevant text entries for a symbol, based on
        combined similarity and importance+recency.

        Args:
            query_text (str): Query string to embed.
            top_k (int): Number of results to retrieve.
            symbol (str): Which symbol to query.

        Returns:
            (List[str], List[int]): The top_k memory texts and their IDs.
        """
        if (
            symbol not in self.universe
            or len(self.universe[symbol]["score_memory"]) == 0
            or top_k <= 0
        ):
            return [], []

        max_len = len(self.universe[symbol]["score_memory"])
        top_k = min(top_k, max_len)
        symbol_index = self.universe[symbol]["index"]

        query_emb = self.embedding_function(query_text)
        # Step 1: top_k by embedding similarity
        p1_dists, p1_ids = symbol_index.search(query_emb, top_k)
        p1_dists, p1_ids = p1_dists[0].tolist(), p1_ids[0].tolist()

        aggregated_texts = []
        aggregated_scores = []
        aggregated_ids = []

        for dist_val, rec_id in zip(p1_dists, p1_ids):
            cur_record = next(
                (r for r in self.universe[symbol]["score_memory"] if r["id"] == rec_id),
                None
            )
            if cur_record is not None:
                aggregated_texts.append(cur_record["text"])
                aggregated_ids.append(cur_record["id"])
                combined_score = self.compound_score_calc_func.merge_score(
                    dist_val,
                    cur_record["importance_recency_compound_score"],
                )
                aggregated_scores.append(combined_score)

        # Step 2: also consider top_k items by importance+recency alone
        #        and re-rank them by combining with embedding similarity
        p2_ids = [
            self.universe[symbol]["score_memory"][i]["id"]
            for i in range(top_k)
        ]
        p2_embeddings = np.vstack(
            [symbol_index.reconstruct(i) for i in p2_ids]
        )
        temp_index = faiss.IndexFlatIP(self.embedding_dim)
        temp_index = faiss.IndexIDMap2(temp_index)
        temp_index.add_with_ids(p2_embeddings, np.array(p2_ids))

        p2_dists, p2_ids = temp_index.search(query_emb, top_k)
        p2_dists, p2_ids = p2_dists[0].tolist(), p2_ids[0].tolist()

        for dist_val, rec_id in zip(p2_dists, p2_ids):
            cur_record = next(
                (r for r in self.universe[symbol]["score_memory"] if r["id"] == rec_id),
                None
            )
            if cur_record is not None:
                combined_score = self.compound_score_calc_func.merge_score(
                    dist_val,
                    cur_record["importance_recency_compound_score"]
                )
                aggregated_texts.append(cur_record["text"])
                aggregated_scores.append(combined_score)
                aggregated_ids.append(cur_record["id"])

        # Rank final aggregated results by combined_score
        score_rank = np.argsort(aggregated_scores)[::-1][:top_k]
        ranked_texts = [aggregated_texts[i] for i in score_rank]
        ranked_ids = [aggregated_ids[i] for i in score_rank]

        # Remove duplicates while preserving order
        seen = set()
        final_texts = []
        final_ids = []
        for t, idx in zip(ranked_texts, ranked_ids):
            if idx not in seen:
                final_texts.append(t)
                final_ids.append(idx)
                seen.add(idx)

        return final_texts, final_ids

    def update_access_count_with_feed_back(
        self,
        symbol: str,
        ids: List[int],
        feedback: List[int]
    ) -> List[int]:
        """
        Update records for a given symbol with new access counts
        (reinforcement feedback).

        Args:
            symbol (str): The symbol to update.
            ids (List[int]): The record IDs to update.
            feedback (List[int]): Feedback values for each ID.

        Returns:
            List[int]: The IDs successfully updated.
        """
        if symbol not in self.universe:
            return []
        success_ids = []
        score_memory = self.universe[symbol]["score_memory"]
        for rec_id, fb in zip(ids, feedback):
            for record in score_memory:
                if record["id"] == rec_id:
                    record["access_counter"] += fb
                    record["importance_score"] = self.importance_score_change_access_counter(
                        access_counter=record["access_counter"],
                        importance_score=record["importance_score"],
                    )
                    record["importance_recency_compound_score"] = self.compound_score_calc_func.recency_and_importance_score(
                        recency_score=record["recency_score"],
                        importance_score=record["importance_score"]
                    )
                    success_ids.append(rec_id)
                    break
        return success_ids

    def _decay(self) -> None:
        """
        Apply the decay function to each record's importance score and recency score.
        """
        for symbol in self.universe:
            score_mem = self.universe[symbol]["score_memory"]
            for i in range(len(score_mem)):
                old_importance = score_mem[i]["importance_score"]
                old_delta = score_mem[i]["delta"]
                new_recency, new_importance, new_delta = self.decay_function(
                    important_score=old_importance,
                    delta=old_delta,
                )
                score_mem[i]["recency_score"] = new_recency
                score_mem[i]["importance_score"] = new_importance
                score_mem[i]["delta"] = new_delta
                score_mem[i]["importance_recency_compound_score"] = \
                    self.compound_score_calc_func.recency_and_importance_score(
                        recency_score=new_recency,
                        importance_score=new_importance,
                    )
            self.universe[symbol]["score_memory"] = score_mem

    def _clean_up(self) -> List[int]:
        """
        Remove records whose recency or importance fall below certain thresholds.

        Returns:
            List[int]: IDs of removed records.
        """
        removed_ids = []
        for symbol in self.universe:
            score_mem = self.universe[symbol]["score_memory"]
            to_remove = [
                mem["id"]
                for mem in score_mem
                if mem["recency_score"] < self.clean_up_threshold_dict["recency_threshold"]
                or mem["importance_score"] < self.clean_up_threshold_dict["importance_threshold"]
            ]
            if to_remove:
                new_list = SortedList(
                    [r for r in score_mem if r["id"] not in to_remove],
                    key=lambda x: x["importance_recency_compound_score"]
                )
                self.universe[symbol]["score_memory"] = new_list
                self.universe[symbol]["index"].remove_ids(np.array(to_remove))
                removed_ids.extend(to_remove)
        return removed_ids

    def step(self) -> List[int]:
        """
        Update memory each step: apply decay, then clean up.

        Returns:
            List[int]: IDs of removed records.
        """
        self._decay()
        removed = self._clean_up()
        return removed

    def prepare_jump(
        self,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], List[int]]:
        """
        Find records that cross thresholds (jump up or jump down).

        Returns:
            Tuple of:
             - jump_dict_up: { symbol -> {"jump_object_list": [...], "emb_list": [...]} }
             - jump_dict_down: { symbol -> {"jump_object_list": [...], "emb_list": [...]} }
             - removed_ids: IDs of the records that were removed from this memory after jumping.
        """
        jump_dict_up: Dict[str, Dict[str, Any]] = {}
        jump_dict_down: Dict[str, Dict[str, Any]] = {}
        removed_ids: List[int] = []

        for symbol in self.universe:
            sym_score_mem = self.universe[symbol]["score_memory"]
            to_delete_up = []
            to_delete_down = []
            jump_list_up = []
            jump_list_down = []
            emb_list_up = []
            emb_list_down = []

            # Identify candidates
            for record in sym_score_mem:
                if record["importance_score"] >= self.jump_threshold_upper:
                    to_delete_up.append(record["id"])
                    jump_list_up.append(record)
                    emb_list_up.append(
                        self.universe[symbol]["index"].reconstruct(record["id"])
                    )
                elif record["importance_score"] < self.jump_threshold_lower:
                    to_delete_down.append(record["id"])
                    jump_list_down.append(record)
                    emb_list_down.append(
                        self.universe[symbol]["index"].reconstruct(record["id"])
                    )

            symbol_to_delete = to_delete_up + to_delete_down
            removed_ids.extend(symbol_to_delete)
            self.universe[symbol]["index"].remove_ids(np.array(symbol_to_delete))

            new_mem = SortedList(
                (r for r in sym_score_mem if r["id"] not in symbol_to_delete),
                key=lambda x: x["importance_recency_compound_score"]
            )
            self.universe[symbol]["score_memory"] = new_mem

            if jump_list_up:
                jump_dict_up[symbol] = {
                    "jump_object_list": jump_list_up,
                    "emb_list": np.vstack(emb_list_up)
                }
            if jump_list_down:
                jump_dict_down[symbol] = {
                    "jump_object_list": jump_list_down,
                    "emb_list": np.vstack(emb_list_down)
                }

        return jump_dict_up, jump_dict_down, removed_ids

    def accept_jump(self, jump_dict: Dict[str, Dict[str, Any]], direction: str) -> None:
        """
        Accept jump records from another memory layer.

        Args:
            jump_dict (dict): A dict containing the records to jump.
            direction (str): "up" (to a higher layer) or "down" (to a lower layer).
        """
        if direction not in ["up", "down"]:
            raise ValueError("direction must be either 'up' or 'down'")

        for symbol, data in jump_dict.items():
            if symbol not in self.universe:
                self.add_new_symbol(symbol)

            new_ids = []
            # If going up, reset recency?
            for rec in data["jump_object_list"]:
                new_ids.append(rec["id"])
                if direction == "up":
                    rec["recency_score"] = self.recency_score_init_func()
                    rec["delta"] = 0

            self.universe[symbol]["score_memory"].update(data["jump_object_list"])
            self.universe[symbol]["index"].add_with_ids(data["emb_list"], np.array(new_ids))

    def save_checkpoint(self, name: str, path: str, force: bool = False) -> None:
        """
        Save this MemoryDB to disk (Faiss indexes + pickled records).
        """
        db_path = os.path.join(path, name)
        if os.path.exists(db_path):
            if not force:
                raise FileExistsError(f"Memory DB {db_path} already exists.")
            shutil.rmtree(db_path)
        os.mkdir(db_path)

        # Basic state
        state_dict = {
            "db_name": self.db_name,
            "id_generator": self.id_generator,
            "jump_threshold_upper": self.jump_threshold_upper,
            "jump_threshold_lower": self.jump_threshold_lower,
            "embedding_dim": self.embedding_dim,
            "embedding_function": self.embedding_function,
            "importance_score_init_func": self.importance_score_init_func,
            "recency_score_init_func": self.recency_score_init_func,
            "compound_score_calc_func": self.compound_score_calc_func,
            "decay_function": self.decay_function,
            "importance_score_change_access_counter": self.importance_score_change_access_counter,
            "clean_up_threshold_dict": self.clean_up_threshold_dict,
        }
        with open(os.path.join(db_path, "state_dict.pkl"), "wb") as f:
            pickle.dump(state_dict, f)

        # Universe
        save_universe = {}
        for sym, data in self.universe.items():
            faiss_path = os.path.join(db_path, f"{sym}.index")
            faiss.write_index(data["index"], faiss_path)
            save_universe[sym] = {
                "score_memory": list(data["score_memory"]),
                "index_save_path": faiss_path,
            }
        with open(os.path.join(db_path, "universe_index.pkl"), "wb") as f:
            pickle.dump(save_universe, f)

    @classmethod
    def load_checkpoint(cls, path: str) -> "MemoryDB":
        """
        Load a MemoryDB from disk.

        Args:
            path (str): Path to the memory DB folder.

        Returns:
            MemoryDB: The restored memory database.
        """
        with open(os.path.join(path, "state_dict.pkl"), "rb") as f:
            state_dict = pickle.load(f)
        with open(os.path.join(path, "universe_index.pkl"), "rb") as f:
            loaded_universe = pickle.load(f)

        # Rebuild FAISS indexes and sorted memory
        for sym in loaded_universe:
            faiss_index = faiss.read_index(loaded_universe[sym]["index_save_path"])
            loaded_universe[sym]["index"] = faiss_index
            loaded_universe[sym]["score_memory"] = SortedList(
                loaded_universe[sym]["score_memory"],
                key=lambda x: x["importance_recency_compound_score"]
            )
            del loaded_universe[sym]["index_save_path"]

        # Construct the new MemoryDB
        db_instance = cls(
            db_name=state_dict["db_name"],
            id_generator=state_dict["id_generator"],
            jump_threshold_upper=state_dict["jump_threshold_upper"],
            jump_threshold_lower=state_dict["jump_threshold_lower"],
            embedding_function=state_dict["embedding_function"],
            importance_score_initialization=state_dict["importance_score_init_func"],
            recency_score_initialization=state_dict["recency_score_init_func"],
            compound_score_calculation=state_dict["compound_score_calc_func"],
            importance_score_change_access_counter=state_dict[
                "importance_score_change_access_counter"
            ],
            decay_function=state_dict["decay_function"],
            clean_up_threshold_dict=state_dict["clean_up_threshold_dict"],
        )
        db_instance.universe = loaded_universe
        return db_instance


class BrainDB:
    """
    Aggregates four MemoryDB objects (short, mid, long, reflection)
    for a single agent, enabling coordinated memory jumps and decays.
    """

    def __init__(
        self,
        agent_name: str,
        id_generator: IdGeneratorFunc,
        embedding_function: OpenAILongerThanContextEmb,
        short_term_memory: MemoryDB,
        mid_term_memory: MemoryDB,
        long_term_memory: MemoryDB,
        reflection_memory: MemoryDB,
        use_gpu: bool = True,
    ):
        """
        Initialize the BrainDB that holds all memory layers.
        """
        self.agent_name = agent_name
        self.embedding_function = embedding_function
        self.use_gpu = use_gpu
        self.id_generator = id_generator

        self.short_term_memory = short_term_memory
        self.mid_term_memory = mid_term_memory
        self.long_term_memory = long_term_memory
        self.reflection_memory = reflection_memory

        self.removed_ids: List[int] = []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BrainDB":
        """
        Build a BrainDB from a unified config. 
        This is a standard approach for each memory layer (short, mid, long, reflection).
        """
        id_generator = IdGeneratorFunc()
        emb_func = OpenAILongerThanContextEmb(**config["embedding"]["detail"])
        agent_name = config["general"]["agent_name"]

        short_db = MemoryDB(
            db_name=f"{agent_name}_short",
            id_generator=id_generator,
            jump_threshold_upper=config["short"]["jump_threshold_upper"],
            jump_threshold_lower=-999999999,  # short doesn't jump down
            embedding_function=emb_func,
            importance_score_initialization=get_importance_score_initialization_func(
                type=config["short"]["importance_score_initialization"],
                memory_layer="short",
            ),
            recency_score_initialization=R_ConstantInitialization(),
            compound_score_calculation=LinearCompoundScore(),
            importance_score_change_access_counter=LinearImportanceScoreChange(),
            decay_function=ExponentialDecay(**config["short"]["decay_params"]),
            clean_up_threshold_dict=config["short"]["clean_up_threshold_dict"],
        )

        mid_db = MemoryDB(
            db_name=f"{agent_name}_mid",
            id_generator=id_generator,
            jump_threshold_upper=config["mid"]["jump_threshold_upper"],
            jump_threshold_lower=config["mid"]["jump_threshold_lower"],
            embedding_function=emb_func,
            importance_score_initialization=get_importance_score_initialization_func(
                type=config["mid"]["importance_score_initialization"],
                memory_layer="mid",
            ),
            recency_score_initialization=R_ConstantInitialization(),
            compound_score_calculation=LinearCompoundScore(),
            importance_score_change_access_counter=LinearImportanceScoreChange(),
            decay_function=ExponentialDecay(**config["mid"]["decay_params"]),
            clean_up_threshold_dict=config["mid"]["clean_up_threshold_dict"],
        )

        long_db = MemoryDB(
            db_name=f"{agent_name}_long",
            id_generator=id_generator,
            jump_threshold_upper=999999999,  # long doesn't jump up
            jump_threshold_lower=config["long"]["jump_threshold_lower"],
            embedding_function=emb_func,
            importance_score_initialization=get_importance_score_initialization_func(
                type=config["long"]["importance_score_initialization"],
                memory_layer="long",
            ),
            recency_score_initialization=R_ConstantInitialization(),
            compound_score_calculation=LinearCompoundScore(),
            importance_score_change_access_counter=LinearImportanceScoreChange(),
            decay_function=ExponentialDecay(**config["long"]["decay_params"]),
            clean_up_threshold_dict=config["long"]["clean_up_threshold_dict"],
        )

        reflection_db = MemoryDB(
            db_name=f"{agent_name}_reflection",
            id_generator=id_generator,
            jump_threshold_upper=999999999,  # reflection doesn't jump up
            jump_threshold_lower=-999999999,  # reflection doesn't jump down
            embedding_function=emb_func,
            importance_score_initialization=get_importance_score_initialization_func(
                type=config["reflection"]["importance_score_initialization"],
                memory_layer="reflection",
            ),
            recency_score_initialization=R_ConstantInitialization(),
            compound_score_calculation=LinearCompoundScore(),
            importance_score_change_access_counter=LinearImportanceScoreChange(),
            decay_function=ExponentialDecay(**config["reflection"]["decay_params"]),
            clean_up_threshold_dict=config["reflection"]["clean_up_threshold_dict"],
        )

        return cls(
            agent_name=agent_name,
            id_generator=id_generator,
            embedding_function=emb_func,
            short_term_memory=short_db,
            mid_term_memory=mid_db,
            long_term_memory=long_db,
            reflection_memory=reflection_db,
        )

    def add_memory_short(self, symbol: str, date_added: date, text: Union[List[str], str]) -> None:
        """Add short-term memory record."""
        self.short_term_memory.add_memory(symbol, date_added, text)

    def add_memory_mid(self, symbol: str, date_added: date, text: Union[List[str], str]) -> None:
        """Add mid-term memory record."""
        self.mid_term_memory.add_memory(symbol, date_added, text)

    def add_memory_long(self, symbol: str, date_added: date, text: Union[List[str], str]) -> None:
        """Add long-term memory record."""
        self.long_term_memory.add_memory(symbol, date_added, text)

    def add_memory_reflection(self, symbol: str, date_added: date, text: Union[List[str], str]) -> None:
        """Add reflection-level memory record."""
        self.reflection_memory.add_memory(symbol, date_added, text)

    def query_short(self, query_text: str, top_k: int, symbol: str) -> Tuple[List[str], List[int]]:
        """Query short-term memory."""
        return self.short_term_memory.query(query_text, top_k, symbol)

    def query_mid(self, query_text: str, top_k: int, symbol: str) -> Tuple[List[str], List[int]]:
        """Query mid-term memory."""
        return self.mid_term_memory.query(query_text, top_k, symbol)

    def query_long(self, query_text: str, top_k: int, symbol: str) -> Tuple[List[str], List[int]]:
        """Query long-term memory."""
        return self.long_term_memory.query(query_text, top_k, symbol)

    def query_reflection(self, query_text: str, top_k: int, symbol: str) -> Tuple[List[str], List[int]]:
        """Query reflection-level memory."""
        return self.reflection_memory.query(query_text, top_k, symbol)

    def update_access_count_with_feed_back(
        self,
        symbol: str,
        ids: Union[List[int], int],
        feedback: int
    ) -> None:
        """
        Update the access counts for memory records matching the given IDs.
        """
        if isinstance(ids, int):
            ids = [ids]
        # Filter out any IDs that were removed
        ids = [i for i in ids if i not in self.removed_ids]
        feedback_list = list(repeat(feedback, len(ids)))
        updated_ids = []

        # Try short layer
        short_updated = self.short_term_memory.update_access_count_with_feed_back(
            symbol, ids, feedback_list
        )
        updated_ids.extend(short_updated)

        # remove updated
        ids = [x for x in ids if x not in short_updated]
        feedback_list = list(repeat(feedback, len(ids)))

        # Mid layer
        if ids:
            mid_updated = self.mid_term_memory.update_access_count_with_feed_back(
                symbol, ids, feedback_list
            )
            updated_ids.extend(mid_updated)
            ids = [x for x in ids if x not in mid_updated]
            feedback_list = list(repeat(feedback, len(ids)))

        # Long layer
        if ids:
            long_updated = self.long_term_memory.update_access_count_with_feed_back(
                symbol, ids, feedback_list
            )
            updated_ids.extend(long_updated)
            ids = [x for x in ids if x not in long_updated]
            feedback_list = list(repeat(feedback, len(ids)))

        # Reflection layer
        if ids:
            reflection_updated = self.reflection_memory.update_access_count_with_feed_back(
                symbol, ids, feedback_list
            )
            updated_ids.extend(reflection_updated)

    def step(self) -> None:
        """
        Advance all memory layers one step (decay, clean up).
        Then handle memory jumps: short->mid, mid->long, mid->short, long->mid, etc.
        """
        # 1. Decay + Cleanup
        self.removed_ids.extend(self.short_term_memory.step())
        for sym in self.short_term_memory.universe:
            logger.info(f"ShortTerm memory for {sym}:")
            for rec in self.short_term_memory.universe[sym]["score_memory"]:
                logger.info(f"{rec}")

        self.removed_ids.extend(self.mid_term_memory.step())
        for sym in self.mid_term_memory.universe:
            logger.info(f"MidTerm memory for {sym}:")
            for rec in self.mid_term_memory.universe[sym]["score_memory"]:
                logger.info(f"{rec}")

        self.removed_ids.extend(self.long_term_memory.step())
        for sym in self.long_term_memory.universe:
            logger.info(f"LongTerm memory for {sym}:")
            for rec in self.long_term_memory.universe[sym]["score_memory"]:
                logger.info(f"{rec}")

        self.removed_ids.extend(self.reflection_memory.step())
        for sym in self.reflection_memory.universe:
            logger.info(f"ReflectionTerm memory for {sym}:")
            for rec in self.reflection_memory.universe[sym]["score_memory"]:
                logger.info(f"{rec}")

        # 2. Perform memory jumps
        logger.info("Starting memory jump cycle...")
        for _ in range(2):
            # short => mid
            logger.info("Processing short-term memory jump ...")
            up_dict, down_dict, removed = self.short_term_memory.prepare_jump()
            self.removed_ids.extend(removed)

            # Accept "up" jumps into mid, "down" is empty for short
            self.mid_term_memory.accept_jump(up_dict, "up")
            # For short, a "down" jump doesn't make sense, but we capture it for logs:
            for sym in up_dict:
                logger.info(f"Up-jump from short => mid for {sym}: {up_dict[sym]['jump_object_list']}")
            for sym in down_dict:
                logger.info(f"Down-jump from short => ??? for {sym}: {down_dict[sym]['jump_object_list']}")

            logger.info("Short => Mid memory jump done.")

            # mid => long + short
            logger.info("Processing mid-term memory jump ...")
            up_dict, down_dict, removed = self.mid_term_memory.prepare_jump()
            self.removed_ids.extend(removed)

            self.long_term_memory.accept_jump(up_dict, "up")
            self.short_term_memory.accept_jump(down_dict, "down")

            for sym in up_dict:
                logger.info(f"Up-jump from mid => long for {sym}: {up_dict[sym]['jump_object_list']}")
            for sym in down_dict:
                logger.info(f"Down-jump from mid => short for {sym}: {down_dict[sym]['jump_object_list']}")

            logger.info("Mid => Long + Short memory jump done.")

            # long => mid
            logger.info("Processing long-term memory jump ...")
            up_dict, down_dict, removed = self.long_term_memory.prepare_jump()
            self.removed_ids.extend(removed)

            # For long, "up" doesn't happen. "down" => mid
            self.mid_term_memory.accept_jump(down_dict, "down")

            for sym in up_dict:
                logger.info(f"Up-jump from long => ??? for {sym}: {up_dict[sym]['jump_object_list']}")
            for sym in down_dict:
                logger.info(f"Down-jump from long => mid for {sym}: {down_dict[sym]['jump_object_list']}")

            logger.info("Long => Mid memory jump done.")

        logger.info("Memory jump cycle complete.")

    def save_checkpoint(self, path: str, force: bool = False) -> None:
        """
        Save the entire BrainDB to disk, including the four memory layers.

        Args:
            path (str): Directory to create.
            force (bool): Whether to overwrite existing directory.
        """
        if os.path.exists(path):
            if not force:
                raise FileExistsError(f"Brain DB path {path} already exists.")
            shutil.rmtree(path)
        os.mkdir(path)

        state_dict = {
            "agent_name": self.agent_name,
            "use_gpu": self.use_gpu,
            "emb_func": self.embedding_function,
            "id_generator": self.id_generator,
            "removed_ids": self.removed_ids,
        }

        with open(os.path.join(path, "state_dict.pkl"), "wb") as f:
            pickle.dump(state_dict, f)

        self.short_term_memory.save_checkpoint("short_term_memory", path, force=force)
        self.mid_term_memory.save_checkpoint("mid_term_memory", path, force=force)
        self.long_term_memory.save_checkpoint("long_term_memory", path, force=force)
        self.reflection_memory.save_checkpoint("reflection_memory", path, force=force)

    @classmethod
    def load_checkpoint(cls, path: str) -> "BrainDB":
        """
        Load BrainDB state from disk, including all memory layers.

        Args:
            path (str): The path where the BrainDB is saved.

        Returns:
            BrainDB: A restored BrainDB object.
        """
        with open(os.path.join(path, "state_dict.pkl"), "rb") as f:
            state_dict = pickle.load(f)

        short_db = MemoryDB.load_checkpoint(os.path.join(path, "short_term_memory"))
        mid_db = MemoryDB.load_checkpoint(os.path.join(path, "mid_term_memory"))
        long_db = MemoryDB.load_checkpoint(os.path.join(path, "long_term_memory"))
        reflection_db = MemoryDB.load_checkpoint(os.path.join(path, "reflection_memory"))

        new_instance = cls(
            agent_name=state_dict["agent_name"],
            id_generator=state_dict["id_generator"],
            embedding_function=state_dict["emb_func"],
            short_term_memory=short_db,
            mid_term_memory=mid_db,
            long_term_memory=long_db,
            reflection_memory=reflection_db,
            use_gpu=state_dict["use_gpu"],
        )
        new_instance.removed_ids = state_dict["removed_ids"]
        return new_instance
