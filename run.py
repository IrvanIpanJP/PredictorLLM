import os
import pickle
import logging
import warnings
from datetime import datetime

import openai
import toml
import typer
from dotenv import load_dotenv
from tqdm import tqdm

from puppy import MarketEnvironment, LLMAgent, RunMode

# -------------------------------------------------------------------
# INITIAL SETUP
# -------------------------------------------------------------------

# Load environment variables from a .env file, if present
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Typer CLI
app = typer.Typer(name="puppy")

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
file_handler = logging.FileHandler("run.log", mode="a")
file_handler.setFormatter(logging_formatter)
logger.addHandler(file_handler)

# -------------------------------------------------------------------
# HELPER FUNCTION
# -------------------------------------------------------------------

def verify_run_mode(mode: str) -> RunMode:
    """
    Verify the provided run mode is 'train' or 'test' and return the enum.

    :param mode: 'train' or 'test'
    :return: RunMode.Train or RunMode.Test
    :raises ValueError: if the mode is invalid.
    """
    if mode == "train":
        return RunMode.Train
    elif mode == "test":
        return RunMode.Test
    raise ValueError("run_mode must be either 'train' or 'test'.")


# -------------------------------------------------------------------
# COMMAND 1: SIMULATION
# -------------------------------------------------------------------

@app.command("sim", help="Start a fresh simulation.")
def simulate(
    market_data_info_path: str = typer.Option(
        os.path.join("data", "06_input", "subset_symbols.pkl"),
        "--market-data-path", "-mdp",
        help="Path to the environment data pickle.",
    ),
    start_time: str = typer.Option(
        "2022-04-04", "--start-time", "-st",
        help="Start date (YYYY-MM-DD).",
    ),
    end_time: str = typer.Option(
        "2022-06-15", "--end-time", "-et",
        help="End date (YYYY-MM-DD).",
    ),
    run_mode: str = typer.Option(
        "train", "--run-mode", "-rm",
        help="Run mode: 'train' or 'test'.",
    ),
    config_path: str = typer.Option(
        os.path.join("config", "config.toml"),
        "--config-path", "-cp",
        help="Path to the TOML config file.",
    ),
    checkpoint_path: str = typer.Option(
        os.path.join("data", "09_checkpoint"),
        "--checkpoint-path", "-ckp",
        help="Directory to store checkpoints (agent & environment).",
    ),
    result_path: str = typer.Option(
        os.path.join("data", "11_train_result"),
        "--result-path", "-rp",
        help="Directory to store final results after simulation completes.",
    )
) -> None:
    """
    Starts a brand-new simulation using the specified config, environment data,
    and run mode (train/test).
    """
    # Validate run mode
    run_mode_var = verify_run_mode(run_mode)

    # Load config
    config = toml.load(config_path)
    logger.info(f"Loaded config from {config_path}.")

    # Load environment data
    if not os.path.exists(market_data_info_path):
        logger.error(f"Market data file not found: {market_data_info_path}")
        raise FileNotFoundError(market_data_info_path)
    with open(market_data_info_path, "rb") as f:
        env_data_pkl = pickle.load(f)
    logger.info(f"Loaded environment data from {market_data_info_path}.")

    # Initialize MarketEnvironment
    environment = MarketEnvironment(
        symbol=config["general"]["trading_symbol"],
        env_data_pkl=env_data_pkl,
        start_date=datetime.strptime(start_time, "%Y-%m-%d").date(),
        end_date=datetime.strptime(end_time, "%Y-%m-%d").date(),
    )

    # Create the LLM agent from config
    the_agent = LLMAgent.from_config(config)
    logger.info("Created LLMAgent from config.")

    # Simulation loop
    pbar = tqdm(total=environment.simulation_length, desc="Simulation Progress")
    while True:
        logger.info(f"Starting step {the_agent.counter}.")
        the_agent.counter += 1

        # Pull next market info
        market_info = environment.step()
        if market_info[-1]:  # 'done' flag is True
            logger.info("Environment returned 'done' flag; ending simulation.")
            break

        current_date = market_info[0]
        logger.info(f"Processing date: {current_date}")
        logger.info(f"Record (future diff): {market_info[-2]}")

        # Step the agent
        the_agent.step(market_info=market_info, run_mode=run_mode_var)  # type: ignore

        # Update progress bar
        pbar.update(1)

        # Checkpoint every step (can be adapted if needed)
        the_agent.save_checkpoint(path=checkpoint_path, force=True)
        environment.save_checkpoint(path=checkpoint_path, force=True)

    # Simulation completed
    pbar.close()
    logger.info("Simulation completed.")

    # Save final results
    logger.info(f"Saving final results to {result_path}...")
    the_agent.save_checkpoint(path=result_path, force=True)
    environment.save_checkpoint(path=result_path, force=True)
    logger.info("All final artifacts saved.")


# -------------------------------------------------------------------
# COMMAND 2: SIM-CHECKPOINT
# -------------------------------------------------------------------

@app.command("sim-checkpoint", help="Resume simulation from an existing checkpoint.")
def simulate_checkpoint(
    checkpoint_path: str = typer.Option(
        os.path.join("data", "09_checkpoint"),
        "--checkpoint-path", "-cp",
        help="Directory with existing environment/agent checkpoints.",
    ),
    result_path: str = typer.Option(
        os.path.join("data", "11_train_result"),
        "--result-path", "-rp",
        help="Directory to store final results after simulation completes.",
    ),
    run_mode: str = typer.Option(
        "train", "--run-mode", "-rm",
        help="Run mode: 'train' or 'test'.",
    )
) -> None:
    """
    Resumes a previous simulation or training run from saved checkpoints. The
    environment and agent state must exist in 'checkpoint_path'.
    """
    # Validate run mode
    run_mode_var = verify_run_mode(run_mode)

    # Load environment checkpoint
    env_checkpoint = os.path.join(checkpoint_path, "env")
    if not os.path.exists(env_checkpoint):
        logger.error(f"Environment checkpoint path not found: {env_checkpoint}")
        raise FileNotFoundError(env_checkpoint)

    environment = MarketEnvironment.load_checkpoint(path=env_checkpoint)
    logger.info(f"Loaded environment checkpoint from {env_checkpoint}.")

    # Load agent checkpoint
    agent_checkpoint = os.path.join(checkpoint_path, "agent_1")
    if not os.path.exists(agent_checkpoint):
        logger.error(f"Agent checkpoint path not found: {agent_checkpoint}")
        raise FileNotFoundError(agent_checkpoint)

    the_agent = LLMAgent.load_checkpoint(path=agent_checkpoint)
    logger.info(f"Loaded agent checkpoint from {agent_checkpoint}.")

    # Resume simulation
    pbar = tqdm(total=environment.simulation_length, desc="Resuming Simulation")
    while True:
        logger.info(f"Resumed step {the_agent.counter}.")
        the_agent.counter += 1

        market_info = environment.step()
        if market_info[-1]:  # if done
            logger.info("Environment returned 'done' flag; ending simulation.")
            break

        the_agent.step(market_info=market_info, run_mode=run_mode_var)  # type: ignore
        pbar.update(1)

        # Checkpoint again (could be changed to every N steps)
        the_agent.save_checkpoint(path=checkpoint_path, force=True)
        environment.save_checkpoint(path=checkpoint_path, force=True)

    pbar.close()
    logger.info("Simulation resumed and completed.")

    # Save final results
    logger.info(f"Saving final results to {result_path}...")
    the_agent.save_checkpoint(path=result_path, force=True)
    environment.save_checkpoint(path=result_path, force=True)
    logger.info("All final artifacts saved.")


# -------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------

if __name__ == "__main__":
    app()
