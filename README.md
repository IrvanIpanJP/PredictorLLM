# PredictorLLM: An Advanced LLM-Based Trading Agent for Stocks and Crypto

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

**PredictorLLM** is an advanced trading agent framework built upon large language models (LLMs) to facilitate automated trading across multiple financial markets—ranging from traditional equities (stocks) to digital assets (cryptocurrencies). This framework integrates a layered, human-inspired memory system with intelligent design features, enabling robust decision-making in dynamic markets.

## Overview

PredictorLLM is composed of three core modules:

1. **Profiling Module**: Establishes agent characteristics, operational scope, and risk preferences (adaptable for both stocks and crypto).
2. **Layered Memory Module**: A structured multi-level memory system, which stores and retrieves relevant textual data (e.g., news, filings, protocol updates) to guide the agent’s trading logic.
3. **Decision-Making Module**: Translates insights from memory into actionable trading strategies. The agent can handle short-term, mid-term, long-term, and reflection-level analyses, mimicking professional traders yet surpassing human limitations in data processing.

By continuously learning from market fluctuations—be it stock price changes or on-chain data for crypto—PredictorLLM adapts to volatility and evolving trends, aiming for optimal investment outcomes.

<div align="center">
  <img src="figures/memory_flow.png" alt="Memory Flow" width="600" />
  <br/><em>Illustration of multi-level memory flow</em>
</div>

<div align="center">
  <img src="figures/workflow.png" alt="Workflow" width="600" />
  <br/><em>Overall workflow architecture</em>
</div>

<div align="center">
  <img src="figures/character.png" alt="Character Design" width="600" />
  <br/><em>Character design concept for the AI trader</em>
</div>

---

## Getting Started

Follow these steps to quickly set up **PredictorLLM** and start trading:

### Prerequisites

- **Python 3.10**: Ensure Python 3.10 or higher is installed on your system.
- **Docker** (optional): For an easy and consistent environment setup.

### 1. Clone the Repository

```bash
git clone https://github.com/IrvanIpanJP/predictor-llm.git
cd predictor-llm
```

### 2. Install Dependencies

We recommend using **Poetry** for managing dependencies. First, install Poetry if you haven't already:

```bash
pip install poetry
```

Then, install the dependencies:

```bash
poetry install
```

You may also activate the virtual environment:

```bash
poetry shell
```

Alternatively, use Docker for a containerized setup:

```bash
docker-compose up --build
```

### 3. Configure Your Agent

Before running the agent, edit the `config/config.toml` file to customize your settings (e.g., market data paths, memory thresholds, or LLM endpoints).

### 4. Running the Agent

To run the agent and get help with available commands, use:

```bash
python run.py --help
```

### 5. Training the Model

To train the agent on your data, use:

```bash
python run.py train
```

Specify the required options like market data path and start/end time:

```bash
python run.py train --market-data-path /path/to/data --start-time 2020-01-01 --end-time 2025-01-01
```

### 6. Resume Training from a Checkpoint

If you need to resume training from a checkpoint:

```bash
python run.py train-checkpoint
```

---

## Practical Examples

### Example 1: Basic Stock Trading Setup

1. **Data Source**: Use Yahoo Finance to fetch stock data.
2. **Configuration**: Set `config/config.toml` to point to your local market data folder containing stock price data.
3. **Run the Agent**:

```bash
python run.py train --market-data-path ./data/stocks --start-time 2023-01-01 --end-time 2023-12-31
```

4. **Outcome**: The agent will analyze the data, store relevant information, and generate stock trading strategies based on historical trends.

### Example 2: Crypto Market Analysis

1. **Data Source**: Fetch cryptocurrency data from CoinGecko using their API.
2. **Configuration**: Point to the crypto data feed in the `config/config.toml` file.
3. **Run the Agent**:

```bash
python run.py train --market-data-path ./data/crypto --start-time 2023-01-01 --end-time 2023-12-31
```

4. **Outcome**: The agent will analyze crypto market behavior, using both price data and on-chain analytics to inform trading strategies.

---

## Notes

### Multi-Asset Data Handling

PredictorLLM supports both **equity** (stock) and **cryptocurrency** data. You can configure custom data sources such as company filings, whitepapers, or on-chain metrics for crypto. The agent stores information in a multi-level memory system, ensuring dynamic decision-making.

### Data Sources

| Type                         | Source                                                | Notes                           | Example Download / API                                       |
| ---------------------------- | ----------------------------------------------------- | ------------------------------- | ------------------------------------------------------------ |
| **Daily Stock Price**        | [Yahoo Finance](https://finance.yahoo.com/)           | Open, High, Low, Close, Volume  | [yfinance](https://pypi.org/project/yfinance/)               |
| **Daily Market News**        | [Alpaca Market News API](https://alpaca.markets/)     | Historical news                 | [Alpaca News API](https://docs.alpaca.markets/docs/news-api) |
| **Company 10-K / 10-Q**      | [SEC EDGAR](https://www.sec.gov/edgar.shtml)          | e.g., Item 7 / Part 1 Item 2    | [SEC API](https://sec-api.io/docs)                           |
| **Crypto Price Feeds**       | [CoinGecko API](https://www.coingecko.com/en/api)     | Daily OHLC data, volume, etc.   | [CoinGecko data endpoints](https://www.coingecko.com/en/api) |
| **On-Chain / Protocol Data** | Etherscan, Polygonscan, or other blockchain explorers | Transaction metrics, DEX volume | Various explorer APIs                                        |

---

## Contributing

Contributions and pull requests to extend crypto-specific features, add additional data sources, or refine memory mechanisms are welcome. Please review our [contribution guidelines](CONTRIBUTING.md) (coming soon).

## License

This project is licensed under the [MIT License](LICENSE).

---

**Happy Trading!**

Leverage the power of LLMs for **stocks**, **crypto**, and beyond — adapt and integrate data sources, and enjoy a flexible, human-like memory system for advanced automated trading.

---

This structure provides an introduction, setup guide, and clear usage instructions with concrete examples for getting started. Let me know if you'd like to adjust anything further!
