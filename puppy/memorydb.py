import os
import faiss
import pickle
import logging
import shutil
import numpy as np
from datetime import date
from itertools import repeat
from sortedcontainers import SortedList
from .embedding import OpenAILongerThanContextEmb
from typing import List, Union, Dict, Any, Tuple, Callable
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
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler = logging.FileHandler("run.log", mode="a")
file_handler.setFormatter(logging_formatter)
logger.addHandler(file_handler)


class id_generator_func:
    def __init__(self):
        self.current_id = 0

    def __call__(self):
        self.current_id += 1
        return self.current_id - 1


class MemoryDB:
    """
    Low-level data store: vector index for text, plus each record’s 
    importance & recency scores, etc. Works for any 'symbol': 
    stock ticker, crypto pair, or other asset identifier.
    """

    def __init__(
        self,
        db_name: str,
        id_generator: Callable,
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
        self.db_name = db_name
        self.id_generator = id_generator
        self.jump_threshold_upper = jump_threshold_upper
        self.jump_threshold_lower = jump_threshold_lower
        self.emb_dim = embedding_function.get_embedding_dimension()
        self.emb_func = embedding_function
        self.importance_score_initialization_func = importance_score_initialization
        self.recency_score_initialization_func = recency_score_initialization
        self.compound_score_calculation_func = compound_score_calculation
        self.decay_function = decay_function
        self.importance_score_change_access_counter = importance_score_change_access_counter
        self.clean_up_threshold_dict = dict(clean_up_threshold_dict)
        self.universe = {}

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], agent_name: str, memory_layer: str
    ) -> "MemoryDB":
        return cls(
            db_name=f"{agent_name}_{memory_layer}",
            id_generator=id_generator_func(),
            jump_threshold_upper=config[memory_layer]["jump_threshold_upper"],
            jump_threshold_lower=config[memory_layer]["jump_threshold_lower"],
            embedding_function=OpenAILongerThanContextEmb(**config["embedding"]),
            importance_score_initialization=get_importance_score_initialization_func(
                type=config[memory_layer]["importance_score_initialization_type"],
                memory_layer=memory_layer,
            ),
            recency_score_initialization=R_ConstantInitialization(),
            compound_score_calculation=LinearCompoundScore(),
            decay_function=ExponentialDecay(
                **config[memory_layer]["decay_params"],
            ),
            importance_score_change_access_counter=LinearImportanceScoreChange(),
            clean_up_threshold_dict=config[memory_layer]["clean_up_threshold_dict"],
        )

    def add_new_symbol(self, symbol: str) -> None:
        cur_index = faiss.IndexFlatIP(self.emb_dim)
        cur_index = faiss.IndexIDMap2(cur_index)
        temp_record = {
            "score_memory": SortedList(
                key=lambda x: x["important_score_recency_compound_score"]
            ),
            "index": cur_index,
        }
        self.universe[symbol] = temp_record

    def add_memory(self, symbol: str, date: date, text: Union[List[str], str]) -> None:
        if symbol not in self.universe:
            self.add_new_symbol(symbol)

        if isinstance(text, str):
            text = [text]

        emb = self.emb_func(text)
        faiss.normalize_L2(emb)
        ids = [self.id_generator() for _ in range(len(text))]

        importance_scores = [
            self.importance_score_initialization_func() for _ in range(len(text))
        ]
        recency_scores = [
            self.recency_score_initialization_func() for _ in range(len(text))
        ]
        partial_scores = [
            self.compound_score_calculation_func.recency_and_importance_score(
                recency_score=r_s, importance_score=i_s
            )
            for i_s, r_s in zip(importance_scores, recency_scores)
        ]
        self.universe[symbol]["index"].add_with_ids(emb, np.array(ids))
        for i in range(len(text)):
            self.universe[symbol]["score_memory"].add(
                {
                    "text": text[i],
                    "id": ids[i],
                    "important_score": importance_scores[i],
                    "recency_score": recency_scores[i],
                    "delta": 0,
                    "important_score_recency_compound_score": partial_scores[i],
                    "access_counter": 0,
                    "date": date,
                }
            )
            logger.info(
                {
                    "text": text[i],
                    "id": ids[i],
                    "important_score": importance_scores[i],
                    "recency_score": recency_scores[i],
                    "delta": 0,
                    "important_score_recency_compound_score": partial_scores[i],
                    "access_counter": 0,
                    "date": date,
                }
            )

    def query(
        self, query_text: str, top_k: int, symbol: str
    ) -> Tuple[List[str], List[int]]:
        if (
            (symbol not in self.universe)
            or (len(self.universe[symbol]["score_memory"]) == 0)
            or (top_k == 0)
        ):
            return [], []

        max_len = len(self.universe[symbol]["score_memory"])
        top_k = min(top_k, max_len)
        cur_index = self.universe[symbol]["index"]

        emb = self.emb_func(query_text)
        # Part 1: Top-k by embedding similarity
        p1_dists, p1_ids = cur_index.search(emb, top_k)
        p1_dists, p1_ids = p1_dists[0].tolist(), p1_ids[0].tolist()

        temp_text_list = []
        temp_score = []
        temp_ids = []

        for cur_sim, cur_id in zip(p1_dists, p1_ids):
            cur_record = next(
                (
                    record
                    for record in self.universe[symbol]["score_memory"]
                    if record["id"] == cur_id
                ),
                None,
            )
            if cur_record:
                temp_text_list.append(cur_record["text"])
                temp_ids.append(cur_record["id"])
                # Combine similarity with recency+importance
                combined_score = self.compound_score_calculation_func.merge_score(
                    cur_sim,
                    cur_record["important_score_recency_compound_score"],
                )
                temp_score.append(combined_score)

        # Part 2: Top-k by partial compound score alone
        p2_ids = [self.universe[symbol]["score_memory"][i]["id"] for i in range(top_k)]
        temp_arrays = [cur_index.reconstruct(i) for i in p2_ids]
        p2_emb = np.vstack(temp_arrays)
        temp_index = faiss.IndexFlatIP(self.emb_dim)
        temp_index = faiss.IndexIDMap2(temp_index)
        temp_index.add_with_ids(p2_emb, np.array(p2_ids))
        p2_dist, p2_ids = temp_index.search(emb, top_k)
        p2_dist, p2_ids = p2_dist[0].tolist(), p2_ids[0].tolist()

        for cur_sim, cur_id in zip(p2_dist, p2_ids):
            cur_record = next(
                (
                    record
                    for record in self.universe[symbol]["score_memory"]
                    if record["id"] == cur_id
                ),
                None,
            )
            if cur_record:
                # combine
                combined_score = self.compound_score_calculation_func.merge_score(
                    cur_sim,
                    cur_record["important_score_recency_compound_score"],
                )
                temp_text_list.append(cur_record["text"])
                temp_score.append(combined_score)
                temp_ids.append(cur_record["id"])

        # Rank by combined score
        score_rank = np.argsort(temp_score)[::-1][:top_k]
        ret_text_list = [temp_text_list[i] for i in score_rank]
        ret_ids = [temp_ids[i] for i in score_rank]

        # Filter duplicates by ID, preserving first occurrence
        _, unique_index = np.unique(ret_ids, return_index=True)
        unique_index = sorted(unique_index, key=lambda x: -temp_score[x])
        final_text_list = [ret_text_list[i] for i in unique_index]
        final_ids = [ret_ids[i] for i in unique_index]

        return final_text_list, final_ids

    def update_access_count_with_feed_back(
        self, symbol: str, ids: List[int], feedback: List[int]
    ) -> List[int]:
        if symbol not in self.universe:
            return []
        success_ids = []
        cur_score_memory = self.universe[symbol]["score_memory"]
        for cur_id, cur_feedback in zip(ids, feedback):
            for record in cur_score_memory:
                if record["id"] == cur_id:
                    record["access_counter"] += cur_feedback
                    record["important_score"] = self.importance_score_change_access_counter(
                        access_counter=record["access_counter"],
                        importance_score=record["important_score"],
                    )
                    record["important_score_recency_compound_score"] = (
                        self.compound_score_calculation_func.recency_and_importance_score(
                            recency_score=record["recency_score"],
                            importance_score=record["important_score"],
                        )
                    )
                    success_ids.append(cur_id)
                    break
        self.universe[symbol]["score_memory"] = cur_score_memory
        return success_ids

    def _decay(self) -> None:
        for cur_symbol in self.universe:
            cur_score_memory = self.universe[cur_symbol]["score_memory"]
            for i in range(len(cur_score_memory)):
                (
                    cur_score_memory[i]["recency_score"],
                    cur_score_memory[i]["important_score"],
                    cur_score_memory[i]["delta"],
                ) = self.decay_function(
                    important_score=cur_score_memory[i]["important_score"],
                    delta=cur_score_memory[i]["delta"],
                )
                cur_score_memory[i]["important_score_recency_compound_score"] = (
                    self.compound_score_calculation_func.recency_and_importance_score(
                        recency_score=cur_score_memory[i]["recency_score"],
                        importance_score=cur_score_memory[i]["important_score"],
                    )
                )
            self.universe[cur_symbol]["score_memory"] = cur_score_memory

    def _clean_up(self) -> List[int]:
        ret_removed_ids = []
        for cur_symbol in self.universe:
            cur_score_memory = self.universe[cur_symbol]["score_memory"]
            if remove_ids := [
                m["id"]
                for m in cur_score_memory
                if m["recency_score"] < self.clean_up_threshold_dict["recency_threshold"]
                or m["important_score"] < self.clean_up_threshold_dict["importance_threshold"]
            ]:
                new_list = SortedList(
                    [],
                    key=lambda x: x["important_score_recency_compound_score"],
                )
                for obj in cur_score_memory:
                    if obj["id"] not in remove_ids:
                        new_list.add(obj)
                self.universe[cur_symbol]["score_memory"] = new_list
                self.universe[cur_symbol]["index"].remove_ids(np.array(remove_ids))
                ret_removed_ids.extend(remove_ids)
        return ret_removed_ids

    def step(self) -> List[int]:
        self._decay()
        return self._clean_up()

    def prepare_jump(
        self,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], List[int]]:
        jump_dict_up = {}
        jump_dict_down = {}
        id_to_remove = []

        for cur_symbol in self.universe:
            temp_delete_ids_up = []
            temp_jump_object_list_up = []
            temp_emb_list_up = []
            temp_delete_ids_down = []
            temp_jump_object_list_down = []
            temp_emb_list_down = []

            cur_score_memory = self.universe[cur_symbol]["score_memory"]
            for record in cur_score_memory:
                if record["important_score"] >= self.jump_threshold_upper:
                    temp_delete_ids_up.append(record["id"])
                    temp_jump_object_list_up.append(record)
                    temp_emb_list_up.append(
                        self.universe[cur_symbol]["index"].reconstruct(record["id"])
                    )
                if record["important_score"] < self.jump_threshold_lower:
                    temp_delete_ids_down.append(record["id"])
                    temp_jump_object_list_down.append(record)
                    temp_emb_list_down.append(
                        self.universe[cur_symbol]["index"].reconstruct(record["id"])
                    )

            temp_delete_ids = temp_delete_ids_up + temp_delete_ids_down
            id_to_remove.extend(temp_delete_ids)
            self.universe[cur_symbol]["index"].remove_ids(np.array(temp_delete_ids))

            new_memory = SortedList(
                [], key=lambda x: x["important_score_recency_compound_score"]
            )
            for r in cur_score_memory:
                if r["id"] not in temp_delete_ids:
                    new_memory.add(r)
            self.universe[cur_symbol]["score_memory"] = new_memory

            if temp_jump_object_list_up:
                temp_emb_list_up = np.vstack(temp_emb_list_up)
                jump_dict_up[cur_symbol] = {
                    "jump_object_list": temp_jump_object_list_up,
                    "emb_list": temp_emb_list_up,
                }
            if temp_jump_object_list_down:
                temp_emb_list_down = np.vstack(temp_emb_list_down)
                jump_dict_down[cur_symbol] = {
                    "jump_object_list": temp_jump_object_list_down,
                    "emb_list": temp_emb_list_down,
                }

        return jump_dict_up, jump_dict_down, id_to_remove

    def accept_jump(self, jump_dict: Dict[str, Dict[str, Any]], direction: str) -> None:
        if direction not in ["up", "down"]:
            raise ValueError("direction must be either [up] or [down]")

        # pick sub-dict for up or down
        jump_subdict = jump_dict[0] if direction == "up" else jump_dict[1]

        for cur_symbol in jump_subdict:
            if cur_symbol not in self.universe:
                self.add_new_symbol(cur_symbol)
            new_ids = []
            for cur_object in jump_subdict[cur_symbol]["jump_object_list"]:
                new_ids.append(cur_object["id"])
                # if direction == "up", reset recency to 1 etc. if needed
                if direction == "up":
                    cur_object["recency_score"] = self.recency_score_initialization_func()
                    cur_object["delta"] = 0

            self.universe[cur_symbol]["score_memory"].update(
                jump_subdict[cur_symbol]["jump_object_list"]
            )
            self.universe[cur_symbol]["index"].add_with_ids(
                jump_subdict[cur_symbol]["emb_list"], np.array(new_ids)
            )

    def save_checkpoint(self, name: str, path: str, force: bool = False) -> None:
        if os.path.exists(os.path.join(path, name)):
            if not force:
                raise FileExistsError(f"Memory db {name} already exists")
            shutil.rmtree(os.path.join(path, name))
        os.mkdir(os.path.join(path, name))
        state_dict = {
            "db_name": self.db_name,
            "id_generator": self.id_generator,
            "jump_threshold_upper": self.jump_threshold_upper,
            "jump_threshold_lower": self.jump_threshold_lower,
            "emb_dim": self.emb_dim,
            "emb_func": self.emb_func,
            "importance_score_initialization_func": self.importance_score_initialization_func,
            "recency_score_initialization_func": self.recency_score_initialization_func,
            "compound_score_calculation_func": self.compound_score_calculation_func,
            "decay_function": self.decay_function,
            "importance_score_change_access_counter": self.importance_score_change_access_counter,
            "clean_up_threshold_dict": self.clean_up_threshold_dict,
        }
        with open(os.path.join(path, name, "state_dict.pkl"), "wb") as f:
            pickle.dump(state_dict, f)

        save_universe = {}
        for cur_symbol in self.universe:
            cur_record = self.universe[cur_symbol]
            faiss.write_index(
                self.universe[cur_symbol]["index"],
                os.path.join(path, name, f"{cur_symbol}.index"),
            )
            save_universe[cur_symbol] = {
                "score_memory": list(cur_record["score_memory"]),
                "index_save_path": os.path.join(path, name, f"{cur_symbol}.index"),
            }
        with open(os.path.join(path, name, "universe_index.pkl"), "wb") as f:
            pickle.dump(save_universe, f)

    @classmethod
    def load_checkpoint(cls, path: str) -> "MemoryDB":
        with open(os.path.join(path, "state_dict.pkl"), "rb") as f:
            state_dict = pickle.load(f)
        with open(os.path.join(path, "universe_index.pkl"), "rb") as f:
            universe = pickle.load(f)

        for sym in universe:
            universe[sym]["index"] = faiss.read_index(universe[sym]["index_save_path"])
            universe[sym]["score_memory"] = SortedList(
                universe[sym]["score_memory"],
                key=lambda x: x["important_score_recency_compound_score"],
            )
            del universe[sym]["index_save_path"]

        obj = cls(
            db_name=state_dict["db_name"],
            id_generator=state_dict["id_generator"],
            jump_threshold_upper=state_dict["jump_threshold_upper"],
            jump_threshold_lower=state_dict["jump_threshold_lower"],
            embedding_function=state_dict["emb_func"],
            importance_score_initialization=state_dict["importance_score_initialization_func"],
            recency_score_initialization=state_dict["recency_score_initialization_func"],
            compound_score_calculation=state_dict["compound_score_calculation_func"],
            importance_score_change_access_counter=state_dict["importance_score_change_access_counter"],
            decay_function=state_dict["decay_function"],
            clean_up_threshold_dict=state_dict["clean_up_threshold_dict"],
        )
        obj.universe = universe.copy()
        return obj


class BrainDB:
    """
    Higher-level aggregator of short-term, mid-term, long-term, 
    and reflection MemoryDB objects. 
    """

    def __init__(
        self,
        agent_name: str,
        id_generator: id_generator_func,
        embedding_function: OpenAILongerThanContextEmb,
        short_term_memory: MemoryDB,
        mid_term_memory: MemoryDB,
        long_term_memory: MemoryDB,
        reflection_memory: MemoryDB,
        use_gpu: bool = True,
    ):
        self.agent_name = agent_name
        self.embedding_function = embedding_function
        self.use_gpu = use_gpu
        self.id_generator = id_generator
        self.short_term_memory = short_term_memory
        self.mid_term_memory = mid_term_memory
        self.long_term_memory = long_term_memory
        self.reflection_memory = reflection_memory
        self.removed_ids = []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BrainDB":
        id_generator = id_generator_func()
        embedding_function = OpenAILongerThanContextEmb(**config["embedding"]["detail"])
        agent_name = config["general"]["agent_name"]

        short_term_memory = MemoryDB(
            db_name=f"{agent_name}_short",
            id_generator=id_generator,
            jump_threshold_upper=config["short"]["jump_threshold_upper"],
            jump_threshold_lower=-999999999,
            embedding_function=embedding_function,
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

        mid_term_memory = MemoryDB(
            db_name=f"{agent_name}_mid",
            id_generator=id_generator,
            jump_threshold_upper=config["mid"]["jump_threshold_upper"],
            jump_threshold_lower=config["mid"]["jump_threshold_lower"],
            embedding_function=embedding_function,
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

        long_term_memory = MemoryDB(
            db_name=f"{agent_name}_long",
            id_generator=id_generator,
            jump_threshold_upper=999999999,
            jump_threshold_lower=config["long"]["jump_threshold_lower"],
            embedding_function=embedding_function,
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

        reflection_memory = MemoryDB(
            db_name=f"{agent_name}_reflection",
            id_generator=id_generator,
            jump_threshold_upper=999999999,
            jump_threshold_lower=-999999999,
            embedding_function=embedding_function,
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
            embedding_function=embedding_function,
            short_term_memory=short_term_memory,
            mid_term_memory=mid_term_memory,
            long_term_memory=long_term_memory,
            reflection_memory=reflection_memory,
        )

    def add_memory_short(self, symbol: str, date: date, text: Union[List[str], str]) -> None:
        self.short_term_memory.add_memory(symbol, date, text)

    def add_memory_mid(self, symbol: str, date: date, text: Union[List[str], str]) -> None:
        self.mid_term_memory.add_memory(symbol, date, text)

    def add_memory_long(self, symbol: str, date: date, text: Union[List[str], str]) -> None:
        self.long_term_memory.add_memory(symbol, date, text)

    def add_memory_reflection(self, symbol: str, date: date, text: Union[List[str], str]) -> None:
        self.reflection_memory.add_memory(symbol, date, text)

    def query_short(self, query_text: str, top_k: int, symbol: str) -> Tuple[List[str], List[int]]:
        return self.short_term_memory.query(query_text, top_k, symbol)

    def query_mid(self, query_text: str, top_k: int, symbol: str) -> Tuple[List[str], List[int]]:
        return self.mid_term_memory.query(query_text, top_k, symbol)

    def query_long(self, query_text: str, top_k: int, symbol: str) -> Tuple[List[str], List[int]]:
        return self.long_term_memory.query(query_text, top_k, symbol)

    def query_reflection(self, query_text: str, top_k: int, symbol: str) -> Tuple[List[str], List[int]]:
        return self.reflection_memory.query(query_text, top_k, symbol)

    def update_access_count_with_feed_back(
        self, symbol: str, ids: Union[List[int], int], feedback: int
    ) -> None:
        if isinstance(ids, int):
            ids = [ids]
        ids = [i for i in ids if i not in self.removed_ids]
        feedback_list = list(repeat(feedback, len(ids)))
        success_ids = []

        # short
        success_ids.extend(self.short_term_memory.update_access_count_with_feed_back(symbol, ids, feedback_list))
        ids = [i for i in ids if i not in success_ids]
        feedback_list = list(repeat(feedback, len(ids)))

        if not ids:
            return

        # mid
        success_ids.extend(self.mid_term_memory.update_access_count_with_feed_back(symbol, ids, feedback_list))
        ids = [i for i in ids if i not in success_ids]
        feedback_list = list(repeat(feedback, len(ids)))

        if not ids:
            return

        # long
        success_ids.extend(self.long_term_memory.update_access_count_with_feed_back(symbol, ids, feedback_list))
        ids = [i for i in ids if i not in success_ids]
        feedback_list = list(repeat(feedback, len(ids)))

        if not ids:
            return

        # reflection
        success_ids.extend(self.reflection_memory.update_access_count_with_feed_back(symbol, ids, feedback_list))

    def step(self) -> None:
        self.removed_ids.extend(self.short_term_memory.step())
        for cur_symbol in self.short_term_memory.universe:
            logger.info(f"short term memory {cur_symbol}")
            for i in self.short_term_memory.universe[cur_symbol]["score_memory"]:
                logger.info(f"memory: {i}")

        self.removed_ids.extend(self.mid_term_memory.step())
        for cur_symbol in self.mid_term_memory.universe:
            logger.info(f"mid term memory {cur_symbol}")
            for i in self.mid_term_memory.universe[cur_symbol]["score_memory"]:
                logger.info(f"memory: {i}")

        self.removed_ids.extend(self.long_term_memory.step())
        for cur_symbol in self.long_term_memory.universe:
            logger.info(f"long term memory {cur_symbol}")
            for i in self.long_term_memory.universe[cur_symbol]["score_memory"]:
                logger.info(f"memory: {i}")

        self.removed_ids.extend(self.reflection_memory.step())
        for cur_symbol in self.reflection_memory.universe:
            logger.info(f"reflection term memory {cur_symbol}")
            for i in self.reflection_memory.universe[cur_symbol]["score_memory"]:
                logger.info(f"memory: {i}")

        # memory jump cycle
        logger.info("Memory jump starts...")
        for _ in range(2):
            # short => mid
            logger.info("Short term memory starts...")
            (jump_dict_up, jump_dict_down, deleted_ids) = self.short_term_memory.prepare_jump()
            self.removed_ids.extend(deleted_ids)
            jump_dict_short = (jump_dict_up, jump_dict_down)
            self.mid_term_memory.accept_jump(jump_dict_short, "up")
            for cur_symbol in jump_dict_up:
                logger.info(f"up-{cur_symbol}: {jump_dict_up[cur_symbol]['jump_object_list']}")
            for cur_symbol in jump_dict_down:
                logger.info(f"down-{cur_symbol}: {jump_dict_down[cur_symbol]['jump_object_list']}")
            logger.info("Short term memory ends...")

            # mid => long & short
            logger.info("Mid term memory starts...")
            (jump_dict_up, jump_dict_down, deleted_ids) = self.mid_term_memory.prepare_jump()
            self.removed_ids.extend(deleted_ids)
            jump_dict_mid = (jump_dict_up, jump_dict_down)
            self.long_term_memory.accept_jump(jump_dict_mid, "up")
            self.short_term_memory.accept_jump(jump_dict_mid, "down")
            for cur_symbol in jump_dict_up:
                logger.info(f"up-{cur_symbol}: {jump_dict_up[cur_symbol]['jump_object_list']}")
            for cur_symbol in jump_dict_down:
                logger.info(f"down-{cur_symbol}: {jump_dict_down[cur_symbol]['jump_object_list']}")
            logger.info("Mid term memory ends...")

            # long => mid
            logger.info("Long term memory starts...")
            (log_jump_dict_up, log_jump_dict_down, deleted_ids) = self.long_term_memory.prepare_jump()
            self.removed_ids.extend(deleted_ids)
            jump_dict_long = (log_jump_dict_up, log_jump_dict_down)
            self.mid_term_memory.accept_jump(jump_dict_long, "down")
            for cur_symbol in log_jump_dict_up:
                logger.info(f"up-{cur_symbol}: {log_jump_dict_up[cur_symbol]['jump_object_list']}")
            for cur_symbol in log_jump_dict_down:
                logger.info(f"down-{cur_symbol}: {log_jump_dict_down[cur_symbol]['jump_object_list']}")
            logger.info("Long term memory ends...")
        logger.info("Memory jump ends...")

    def save_checkpoint(self, path: str, force: bool = False) -> None:
        if os.path.exists(path):
            if not force:
                raise FileExistsError(f"Brain db {path} already exists")
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

        self.short_term_memory.save_checkpoint(name="short_term_memory", path=path, force=force)
        self.mid_term_memory.save_checkpoint(name="mid_term_memory", path=path, force=force)
        self.long_term_memory.save_checkpoint(name="long_term_memory", path=path, force=force)
        self.reflection_memory.save_checkpoint(name="reflection_memory", path=path, force=force)

    @classmethod
    def load_checkpoint(cls, path: str):
        with open(os.path.join(path, "state_dict.pkl"), "rb") as f:
            state_dict = pickle.load(f)

        short_term_memory = MemoryDB.load_checkpoint(os.path.join(path, "short_term_memory"))
        mid_term_memory = MemoryDB.load_checkpoint(os.path.join(path, "mid_term_memory"))
        long_term_memory = MemoryDB.load_checkpoint(os.path.join(path, "long_term_memory"))
        reflection_memory = MemoryDB.load_checkpoint(os.path.join(path, "reflection_memory"))

        return cls(
            agent_name=state_dict["agent_name"],
            id_generator=state_dict["id_generator"],
            embedding_function=state_dict["emb_func"],
            short_term_memory=short_term_memory,
            mid_term_memory=mid_term_memory,
            long_term_memory=long_term_memory,
            reflection_memory=reflection_memory,
        )
