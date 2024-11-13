# PredictorLLM: Advanced LLM-Based Trading Agent for Stocks & Crypto

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

An all-in-one platform leveraging large language models (LLMs) to automate trading in both **traditional equities** (stocks) and **digital assets** (cryptocurrencies). Built on top of a multi-level memory system and advanced reflection mechanism, **PredictorLLM** aims to emulate sophisticated human-like reasoning to adapt and thrive in rapidly changing markets.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Key Modules](#key-modules)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
   - [Basic Command Examples](#basic-command-examples)
   - [Sample Code Snippet](#sample-code-snippet)
   - [Visualization & Results](#visualization--results)
7. [External Resources & Links](#external-resources--links)
8. [Troubleshooting](#troubleshooting)
9. [License](#license)

---

## Overview

**PredictorLLM** combines three major components into a single agent:

1. **Profiling Module** — Determines the agent’s personality, risk tolerance, and broad objectives.
2. **Layered Memory Module** — Organizes data (e.g., price movements, filings, news events) into short, mid, and long-term memories, as well as a reflection layer for meta-analysis.
3. **Decision-Making Module** — Synthesizes memory insights into actionable strategies, supporting both day-to-day trades and longer-range positions.

---

## Prerequisites

- **Python 3.10 or higher**
  - Ensure you have a stable environment, ideally with virtual environments (e.g., `venv`, `conda`, or **Poetry**).
- **Docker (optional)**
  - For a reproducible environment setup.
- **CPU or GPU**
  - A GPU is recommended for large-scale LLM tasks.
- **Disk Space**
  - Ensure sufficient space for storing data and checkpoint files.

---

## Key Modules

- **`agent.py`**: Main driver for the LLM-based trading logic.
- **`memorydb.py`**: Manages multi-layer text embeddings and memory retrieval.
- **`environment.py`**: Simulates or loads historical market data (stocks, crypto).
- **`portfolio.py`**: Tracks the agent’s holdings, actions, and performance.
- **`chat.py`**: Interfaces with different LLM endpoints (e.g., OpenAI, Together).
- **`memory_functions`**: Contains specialized logic for memory scoring, decay, and updates.

---

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/IrvanIpanJP/predictor-llm.git
   cd predictor-llm
   ```

2. **Install Dependencies**  
   Using **Poetry** is recommended:
   ```bash
   pip install poetry
   poetry install
   poetry shell
   ```
   Or, with **Docker**:
   ```bash
   docker-compose up --build
   ```

---

## Configuration

- Locate the main **configuration file** at `config/config.toml`.
- Adjust paths, memory thresholds, run modes, and LLM endpoints:

  ```toml
  [general]
  agent_name = "PredictorAgent"
  trading_symbol = "AAPL"
  look_back_window_size = 7

  [embedding.detail]
  openai_api_key = "YOUR_OPENAI_KEY"
  chunk_size = 2000
  verbose = false
  ```

- For more advanced usage, you can edit thresholds for short/mid/long/reflection memories under `[short]`, `[mid]`, `[long]`, and `[reflection]`.

---

## Usage

### Basic Command Examples

- **Train the Agent**
  ```bash
  python run.py train --market-data-path data/stocks --start-time 2020-01-01 --end-time 2022-01-01
  ```
- **Test the Agent**
  ```bash
  python run.py test --market-data-path data/stocks --start-time 2022-01-01 --end-time 2023-01-01
  ```
- **Resume from Checkpoint**
  ```bash
  python run.py train-checkpoint --checkpoint-path ./checkpoints/PredictorAgent
  ```

### Sample Code Snippet

Below is an example of how you might integrate new short-term news data into the memory:

```python
from datetime import date
from environment import MarketEnvironment
from agent import LLMAgent
from run_type import RunMode

# Instantiate your environment and agent
env = MarketEnvironment(env_data_pkl=my_data, start_date=date(2022,1,1), end_date=date(2022,12,31), symbol="BTC-USD")
my_agent = LLMAgent.from_config(config_dict)

# Step through environment for training
while True:
    step_info = env.step()
    if step_info[-1]:  # 'done' flag
        break
    my_agent.step(market_info=step_info, run_mode=RunMode.Train)
```

This snippet demonstrates how the agent retrieves daily data from the environment and updates memories accordingly.

### Visualization & Results

We have included a function **`Visualize-results.py`** to help you quickly compare different strategies’ cumulative returns. It is based on [Matplotlib](https://matplotlib.org/) and [pandas](https://pandas.pydata.org/). Any similarities to external scripts are properly attributed; references have been included where relevant to ensure no plagiarism is intended.

> **Note**: The function `get_data()` in `Visualize-results.py` is adapted from publicly available examples. We’ve modified it to integrate with our data pipeline. Please review the references or disclaimers in the codebase for details on usage and compliance.

---

## External Resources & Links

- [Yahoo Finance](https://finance.yahoo.com/) for stock price data and fundamentals.
- [CoinGecko API](https://www.coingecko.com/en/api) for crypto price data.
- [SEC EDGAR](https://www.sec.gov/edgar.shtml) for company filings.
- [Matplotlib](https://matplotlib.org/) for Python plotting.
- [pandas](https://pandas.pydata.org/) for data manipulation.

---

## Troubleshooting

1. **Dependency Conflicts**
   - If you see version conflicts, consider running `poetry update` or re-building your Docker container.
2. **LLM Endpoint Errors**
   - Check your API keys (e.g., `OPENAI_API_KEY`). Make sure they’re valid and have usage credits.
3. **Memory Overflows**
   - Reduce the batch size or use a smaller LLM model. For Docker, allocate more memory.
4. **Checkpoint Not Loading**
   - Ensure the structure of the checkpoint folder matches the expected layout (`state_dict.pkl`, `brain/state_dict.pkl`, etc.).

If you’re stuck, open an issue on GitHub or check our discussion forum.

---

## License

This project is open-sourced under the [MIT License](LICENSE). Commercial or personal usage is permitted; please maintain attribution where possible.

---

**Happy Trading!**  
Use LLMs and a dynamic memory system to optimize trades for both stocks and crypto markets. We appreciate your feedback, issues, and pull requests!

---
