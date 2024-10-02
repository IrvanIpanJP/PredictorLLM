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

## Usage

### Docker Setup

For a quick and consistent environment, we recommend Docker. A sample `Dockerfile` and VSCode development container config (`.devcontainer/devcontainer.json`) are provided.

### Dependencies

PredictorLLM runs on **Python 3.10**. We use [Poetry](https://python-poetry.org/) for dependency management. Install dependencies with:

```bash
poetry config virtualenvs.in-project true  # Optional: keep virtualenv in project folder
poetry install
```

You may also wish to use [pipx](https://pypa.github.io/pipx/) to install Poetry system-wide. Activate the Poetry shell by running `poetry shell` or `source .venv/bin/activate` (if `.venv` is installed locally).

### Running the Code

Use `run.py` as the primary entry point:

```bash
python run.py --help
```

Configuration settings (e.g., memory thresholds, LLM endpoints, or data paths) are specified in `config/config.toml`.

### Training the Model

To train the agent:

```bash
python run.py train
```

Common options include:

```plaintext
--market-data-path  -mdp      TEXT     The market data path
--start-time        -st       TEXT     The start time
--end-time          -et       TEXT     The end time
--config-path       -cp       TEXT     Config file path [default: config/config.toml]
--checkpoint-path   -ckp      TEXT     Where to store checkpoints
--save-every        -se       INTEGER  Frequency (n steps) for saving checkpoints
--result-path       -rp       TEXT     Where to store training results
--help                                 Show this message and exit.
```

Training automatically saves checkpoints. To resume from the latest checkpoint:

```bash
python run.py train-checkpoint
```

---

## Notes

### Multi-Asset Data Handling

PredictorLLM accommodates both **equity** (stock) data and **cryptocurrency** data. You can provide your own market data feeds, protocol updates, or filings (e.g., SEC 10-Ks for stocks, or on-chain analytics for crypto). The framework’s layered memory allows storing and prioritizing text data—such as **company filings**, **crypto project whitepapers**, **news** articles, and **price** information—in short, mid, long, or reflection-term memory.

### Example Data Sources

| Type                         | Source                                                | Notes                           | Example Download / API                                       |
| ---------------------------- | ----------------------------------------------------- | ------------------------------- | ------------------------------------------------------------ |
| **Daily Stock Price**        | [Yahoo Finance](https://finance.yahoo.com/)           | Open, High, Low, Close, Volume  | [yfinance](https://pypi.org/project/yfinance/)               |
| **Daily Market News**        | [Alpaca Market News API](https://alpaca.markets/)     | Historical news                 | [Alpaca News API](https://docs.alpaca.markets/docs/news-api) |
| **Company 10-K / 10-Q**      | [SEC EDGAR](https://www.sec.gov/edgar.shtml)          | e.g., Item 7 / Part 1 Item 2    | [SEC API](https://sec-api.io/docs)                           |
| **Crypto Price Feeds**       | [CoinGecko API](https://www.coingecko.com/en/api)     | Daily OHLC data, volume, etc.   | [CoinGecko data endpoints](https://www.coingecko.com/en/api) |
| **On-Chain / Protocol Data** | Etherscan, Polygonscan, or other blockchain explorers | Transaction metrics, DEX volume | Various explorer APIs                                        |

### Data Schemas

Below are illustrative column schemas; feel free to adapt them to crypto or other assets:

**Daily Price (Stock/Crypto)**
| Column | Type | Notes |
|-----------|----------|-----------------------------------|
| `Date` | datetime | Trading or historical date |
| `Open` | float | Opening price |
| `High` | float | Highest price during the day |
| `Low` | float | Lowest price during the day |
| `Close` | float | Closing price |
| `Volume` | float | Trading volume or on-chain volume |
| `Symbol` | str | Ticker symbol or crypto pair |

**Daily News**
| Column | Type | Notes |
|-----------|----------|---------------------------|
| `Author` | str | - |
| `Content` | str | Main text of the article |
| `DateTime`| datetime | Timestamp of article |
| `Date` | datetime | Adjusted to trading hours |
| `Source` | str | News source, e.g. 'AP' |
| `Summary` | str | One-line summary |
| `Title` | str | Headline |
| `URL` | str | Link to article |
| `Equity` or `Asset`| str | Stock symbol or crypto name |
| `Text` | str | Concatenated title/summary|

**Company 10-K / 10-Q or Crypto Whitepapers**
| Column | Type | Notes |
|----------------|----------|-------------------------------|
| `Document URL` | str | Link to original filing/doc |
| `Content` | str | Extracted text content |
| `Ticker` | str | Stock ticker or crypto symbol |
| `UTC Timestamp`| datetime | Filing time (UTC) |
| `Type` | str | e.g., “10-K”, “Whitepaper” |

---

## Contributing

Contributions and pull requests to extend crypto-specific features, add additional data sources, or refine memory mechanisms are welcome. Please review our [contribution guidelines](CONTRIBUTING.md) (coming soon).

## License

This project is licensed under the [MIT License](LICENSE).

---

**Happy Trading!**

Leverage the power of LLMs for **stocks**, **crypto**, and beyond — adapt and integrate data sources, and enjoy a flexible, human-like memory system for advanced automated trading.
