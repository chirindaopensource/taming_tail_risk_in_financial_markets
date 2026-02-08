# **`README.md`**

# Taming Tail Risk in Financial Markets: Conformal Risk Control for Nonstationary Portfolio VaR

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2602.03903-b31b1b.svg)](https://arxiv.org/abs/2602.03903)
[![Journal](https://img.shields.io/badge/Journal-ArXiv%20Preprint-003366)](https://arxiv.org/abs/2602.03903)
[![Year](https://img.shields.io/badge/Year-2026-purple)](https://github.com/chirindaopensource/taming_tail_risk_in_financial_markets)
[![Discipline](https://img.shields.io/badge/Discipline-Financial%20Econometrics%20%7C%20Risk%20Management-00529B)](https://github.com/chirindaopensource/taming_tail_risk_in_financial_markets)
[![Data Sources](https://img.shields.io/badge/Data-CRSP%20%7C%20WRDS-lightgrey)](https://wrds-www.wharton.upenn.edu/)
[![Core Method](https://img.shields.io/badge/Method-Regime--Weighted%20Conformal%20Prediction-orange)](https://github.com/chirindaopensource/taming_tail_risk_in_financial_markets)
[![Analysis](https://img.shields.io/badge/Analysis-Sequential%20VaR%20Control-red)](https://github.com/chirindaopensource/taming_tail_risk_in_financial_markets)
[![Validation](https://img.shields.io/badge/Validation-Kupiec%20UC%20%7C%20Christoffersen%20CC-green)](https://github.com/chirindaopensource/taming_tail_risk_in_financial_markets)
[![Robustness](https://img.shields.io/badge/Robustness-Bandwidth%20Ablation%20Sweep-yellow)](https://github.com/chirindaopensource/taming_tail_risk_in_financial_markets)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![YAML](https://img.shields.io/badge/YAML-%23CB171E.svg?style=flat&logo=yaml&logoColor=white)](https://yaml.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen)](https://github.com/chirindaopensource/taming_tail_risk_in_financial_markets)

**Repository:** `https://github.com/chirindaopensource/taming_tail_risk_in_financial_markets`

**Owner:** 2026 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2026 paper entitled **"Taming Tail Risk in Financial Markets: Conformal Risk Control for Nonstationary Portfolio VaR"** by:

*   **Marc Schmitt** (University of Oxford)

The project provides a complete, end-to-end computational framework for replicating the paper's findings. It delivers a modular, auditable, and extensible pipeline that executes the entire research workflow: from the ingestion and rigorous validation of CRSP market data to the sequential forecasting of Value-at-Risk (VaR) using Regime-Weighted Conformal (RWC) calibration, culminating in comprehensive backtesting and robustness analysis.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `run_full_study_pipeline`](#key-callable-run_full_study_pipeline)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the analytical framework presented in Schmitt (2026). The core of this repository is the iPython Notebook `taming_tail_risk_in_financial_markets_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings. The pipeline addresses the critical challenge of **sequential risk control** in nonstationary financial markets, where standard VaR models often fail during structural breaks and volatility regimes.

The paper proposes **Regime-Weighted Conformal Risk Control (RWC)**, a model-agnostic framework that wraps arbitrary quantile forecasters to ensure valid sequential risk control. This codebase operationalizes the proposed solution:
-   **Validates** data integrity using strict schema checks and temporal consistency enforcement.
-   **Engineers** regime features ($RV21$, $MAR5$) to capture market volatility and trend states.
-   **Calibrates** VaR bounds using a novel weighting scheme that combines exponential time decay (recency) with kernel-based regime similarity.
-   **Evaluates** performance via rigorous backtesting (Kupiec, Christoffersen) and regime-stratified stability metrics.

## Theoretical Background

The implemented methods combine techniques from Financial Econometrics, Conformal Prediction, and Statistical Learning.

**1. Sequential Risk Control Objective:**
The goal is to construct a one-sided VaR bound $U_t(x_t)$ such that the conditional probability of exceedance is bounded by $\alpha$:
$$ \mathbb{P}(y_t \le U_t(x_t)) \ge 1 - \alpha $$

**2. Regime-Weighted Conformal (RWC):**
RWC calibrates a safety buffer $\hat{c}_t$ from past forecast errors $s_i = y_i - \hat{q}_i$. Weights $w_i(t)$ are assigned based on recency and regime similarity:
$$ w_i(t) \propto \underbrace{\exp(-\lambda(t-i))}_{\text{Recency}} \cdot \underbrace{K_h(z_i, z_t)}_{\text{Regime Similarity}} $$
where $K_h$ is a Gaussian kernel measuring the distance between regime embeddings $z$.

**3. Weighted Quantile Calibration:**
The buffer $\hat{c}_t$ is computed as the weighted $(1-\alpha)$-quantile of the past scores:
$$ \hat{c}_t := Q_{1-\alpha}^{\tilde{w}(t)}(\{s_i\}_{i \in \mathcal{I}_t}) $$

**4. Effective Sample Size (ESS) Safeguard:**
To prevent variance explosion when localizing to rare regimes, the algorithm monitors the effective sample size $n_{\text{eff}}(t)$. If $n_{\text{eff}}(t) < n_{\min}$, it falls back to time-only weighting (TWC).

Below is a diagram which summarizes the proposed approach:

<div align="center">
  <img src="https://github.com/chirindaopensource/taming_tail_risk_in_financial_markets/blob/main/taming_tail_risk_in_financial_markets_ipo_main.png" alt="RWC System Architecture" width="100%">
</div>

## Features

The provided iPython Notebook (`taming_tail_risk_in_financial_markets_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Multi-Task Architecture:** The pipeline is decomposed into 23 distinct, modular tasks, each with its own orchestrator function.
-   **Configuration-Driven Design:** All study parameters (grids, splits, hyperparameters) are managed in an external `config.yaml` file.
-   **Rigorous Data Validation:** A multi-stage validation process checks schema integrity, temporal monotonicity, and return plausibility.
-   **Deterministic Execution:** Enforces reproducibility through seed control, strict causality checks, and frozen parameter sets.
-   **Comprehensive Audit Logging:** Generates detailed logs of every processing step, including invariant checks and benchmark comparisons.
-   **Reproducible Artifacts:** Generates structured results containing raw time-series, aggregated metrics, and robustness sweep data.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Configuration & Validation (Task 1):** Loads and validates the study configuration, enforcing parameter constraints and reproduction modes.
2.  **Data Ingestion & Cleansing (Tasks 2-3):** Validates CRSP schema, enforces strict monotonicity, and handles missingness.
3.  **Loss Construction (Task 4):** Computes portfolio loss $y_t = -r_t^{\text{port}}$ and enforces causality contracts.
4.  **Data Splitting (Task 5):** Partitions data into Train, Validation, and Test sets based on chronological boundaries.
5.  **Feature Engineering (Tasks 6-7):** Computes and standardizes regime features ($RV21$, $MAR5$) using pre-test statistics.
6.  **Base Forecasting (Tasks 9-10):** Generates quantile forecasts using Historical Simulation (HS) and Gradient Boosting (GBDT).
7.  **Conformal Calibration (Tasks 11-14):** Applies SWC, TWC, RWC, and ACI wrappers to calibrate VaR bounds.
8.  **Hyperparameter Tuning (Task 15):** Optimizes parameters ($m, \lambda, h$) on the validation set.
9.  **Execution (Task 16):** Runs the final optimized models on the full test set.
10. **Evaluation (Tasks 17-20):** Computes headline metrics, regime stability, backtests, and weight diagnostics.
11. **Robustness Analysis (Task 22):** Conducts a bandwidth ablation sweep to verify the bias-variance tradeoff.
12. **Final Audit (Task 23):** Verifies methodological invariants and compares results against paper benchmarks.

## Core Components (Notebook Structure)

The notebook is structured as a logical pipeline with modular orchestrator functions for each of the 23 major tasks. All functions are self-contained, fully documented with type hints and docstrings, and designed for professional-grade execution.

## Key Callable: `run_full_study_pipeline`

The project is designed around a single, top-level user-facing interface function:

-   **`run_full_study_pipeline`:** This master orchestrator function runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire computational portion of the project, managing data flow between validation, forecasting, calibration, evaluation, and auditing modules.

## Prerequisites

-   Python 3.9+
-   Core dependencies: `pandas`, `numpy`, `scipy`, `pyyaml`, `scikit-learn`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/taming_tail_risk_in_financial_markets.git
    cd taming_tail_risk_in_financial_markets
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy scipy pyyaml scikit-learn
    ```

## Input Data Structure

The pipeline requires a primary DataFrame `raw_market_data` with the following columns:

1.  **`DATE`**: `datetime64[ns]`. Strictly increasing trading days. No duplicates or missing dates within the trading calendar.
2.  **`VWRETD`**: `float64`. Value-Weighted Return including Distributions (decimal format, e.g., 0.01 = 1%).

*Note: The pipeline includes a synthetic data generator for testing purposes if access to CRSP is unavailable.*

## Usage

The notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell, which demonstrates how to use the top-level `run_full_study_pipeline` orchestrator:

```python
# Final cell of the notebook

# This block serves as the main entry point for the entire project.
if __name__ == '__main__':
    # 1. Load the master configuration from the YAML file.
    config = load_study_configuration("config.yaml")
    
    # 2. Load raw datasets (Example using synthetic generator provided in the notebook)
    # In production, load from CSV/Parquet: pd.read_csv(...)
    raw_market_data = generate_synthetic_crsp_data()

    # 3. Execute the entire replication study.
    artifacts = run_full_study_pipeline(raw_market_data, config)
    
    # 4. Access results
    print(artifacts["audit_report"])
```

## Output Structure

The pipeline returns a dictionary containing:
-   **`main_results`**: A dictionary mapping model IDs (e.g., "HS_RWC") to tuples of `(DataFrame, metrics_dict)`. The DataFrame contains time-series outputs ($y_t, U_t, I_t$), and the metrics dictionary contains aggregated performance stats.
-   **`robustness_results`**: A dictionary containing DataFrames from the bandwidth ablation sweep.
-   **`audit_report`**: A formatted text string summarizing the reproduction fidelity, invariant checks, and benchmark comparisons.

## Project Structure

```
taming_tail_risk_in_financial_markets/
│
├── taming_tail_risk_in_financial_markets_draft.ipynb   # Main implementation notebook
├── config.yaml                                         # Master configuration file
├── requirements.txt                                    # Python package dependencies
│
├── LICENSE                                             # MIT Project License File
└── README.md                                           # This file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can modify study parameters such as:
-   **Risk Objective:** `target_alpha` (e.g., 0.01 or 0.05).
-   **Data Splits:** Date ranges for Train, Validation, and Test periods.
-   **Model Parameters:** Calibration window size ($m$), decay rate ($\lambda$), kernel bandwidth ($h$).
-   **Evaluation:** Number of regime quintiles, backtesting conventions.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## Recommended Extensions

Future extensions could include:
-   **Alternative Base Models:** Integrating GARCH or CAViaR as base forecasters.
-   **Multi-Step Forecasting:** Extending the framework to multi-period VaR horizons.
-   **Alternative Kernels:** Experimenting with different similarity kernels (e.g., Laplacian, Matern).
-   **Real-Time Deployment:** Adapting the pipeline for live streaming data ingestion.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{schmitt2026taming,
  title={Taming Tail Risk in Financial Markets: Conformal Risk Control for Nonstationary Portfolio VaR},
  author={Schmitt, Marc},
  journal={arXiv preprint arXiv:2602.03903},
  year={2026}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2026). Taming Tail Risk in Financial Markets: An Open Source Implementation.
GitHub repository: https://github.com/chirindaopensource/taming_tail_risk_in_financial_markets
```

## Acknowledgments

-   Credit to **Marc Schmitt** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, including **Pandas, NumPy, SciPy, and Scikit-Learn**.

--

*This README was generated based on the structure and content of the `taming_tail_risk_in_financial_markets_draft.ipynb` notebook and follows best practices for research software documentation.*
