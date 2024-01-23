# PredictorLLM: An Advanced LLM-Based Trading Agent with Enhanced Memory and Design Features

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

This repository provides the Python source code for **PredictorLLM**, an advanced trading agent framework built upon large language models (LLMs) with enhanced memory architecture and intelligent design features.

## Overview

PredictorLLM leverages the capabilities of large language models to facilitate automated trading in dynamic financial markets. This framework integrates three core components:

1. **Profiling Module**: Establishes agent characteristics and operational scope.
2. **Layered Memory Module**: Utilizes a structured memory system inspired by human cognitive processes for retaining and prioritizing financial data.
3. **Decision-Making Module**: Converts insights from memory into actionable trading strategies.

With adjustable memory spans and the ability to assimilate hierarchical information, PredictorLLM mimics the behavior of professional traders while surpassing human limitations in data retention and processing. The framework continuously evolves to improve trading decisions and adapts to volatile market conditions, delivering superior investment outcomes.

![Memory Flow](figures/memory_flow.png)
![Workflow](figures/workflow.png)
![Character Design](figures/character.png)

---

## Usage

### Docker Setup

We recommend using Docker for seamless code execution. The Dockerfile is available at [Dockerfile](), along with a development container setup for VSCode at [devcontainer.json]().

### Dependencies

PredictorLLM runs on Python 3.10. Install all required dependencies using [poetry](https://python-poetry.org/):

```bash
poetry config virtualenvs.in-project true  # Optional: Install virtualenv in the project
poetry install
```

We suggest using [pipx](https://pypa.github.io/pipx/) to install poetry. Activate the virtual environment using `poetry shell` or `source .venv/bin/activate` (if virtualenv is installed in the project folder).

### Running the Code

The entry point for the code is `run.py`. Use the following command to view available options:

```bash
python run.py --help
```

Configuration settings are stored in `config/config.toml`.

### Training the Model

To train the model, use:

```bash
python run.py train
```

Default options include:

```plaintext
--market-data-path  -mdp      TEXT     The market data path [default: /workspaces/ArkGPT/data/06_input/subset_symbols.pkl]                                 │
--start-time        -st       TEXT     The start time [default: 2022-03-14]                                                                                │
--end-time          -et       TEXT     The end time [default: 2022-06-27]                                                                                  │
--config-path       -cp       TEXT     Config file path [default: config/config.toml]                                                                                                                                                │
--checkpoint-path   -ckp      TEXT     The checkpoint path [default: data/09_checkpoint]                                                                   │
--save-every        -se       INTEGER  Save every n steps [default: 1]                                                                                     │
--result-path       -rp       TEXT     The result save path [default: data/11_train_result]                                                                │
--help                                 Show this message and exit.
```

Training automatically saves checkpoints to resume progress in case of interruptions. Resume training using:

```bash
python run.py train-checkpoint
```

---

## Notes

### Data Sources

| Type              | Source                 | Notes                          | Download Method                                              |
| ----------------- | ---------------------- | ------------------------------ | ------------------------------------------------------------ |
| Daily Stock Price | Yahoo Finance          | Open, High, Low, Close, Volume | [yfinance](https://pypi.org/project/yfinance/)               |
| Daily Market News | Alpaca Market News API | Historical news                | [Alpaca News API](https://docs.alpaca.markets/docs/news-api) |
| Company 10-K      | SEC EDGAR              | Item 7                         | [SEC API](https://sec-api.io/docs)                           |
| Company 10-Q      | SEC EDGAR              | Part 1 Item 2                  | [SEC API](https://sec-api.io/docs)                           |

### Data Schemas

**Daily Stock Price**
| Column | Type | Notes |
|-----------|---------|---------------------|
| Date | datetime| - |
| Open | float | Opening price |
| High | float | Highest price |
| Low | float | Lowest price |
| Close | float | Closing price |
| Adj Close | float | Adjusted closing price |
| Volume | float | Trade volume |
| Symbol | str | Ticker symbol |

**Daily Market News**
| Column | Type | Notes |
|-----------|---------|---------------------|
| Author | str | - |
| Content | str | Content of news |
| DateTime | datetime| News timestamp |
| Date | datetime| Adjusted for trading hours |
| Source | str | News source |
| Summary | str | News summary |
| Title | str | News title |
| URL | str | News link |
| Equity | str | Ticker symbol |
| Text | str | Combined title and summary |

**Company 10-K & 10-Q**
| Column | Type | Notes |
|----------------|---------|----------------------------|
| Document URL | str | Link to EDGAR file |
| Content | str | Extracted text content |
| Ticker | str | Company ticker symbol |
| UTC Timestamp | datetime| Coordinated Universal Time |
| EST Timestamp | datetime| Eastern Standard Time |
| Type | str | Report type ("10-K" or "10-Q") |

```

This revision avoids referencing the scientific paper and adjusts the language for general documentation purposes.
```

```

```
