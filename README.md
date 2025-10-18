# MARL-SmartGrid-Pricing

A research project implementing **Multi-Agent Reinforcement Learning (MARL)** for designing adaptive **pricing** and **incentive strategies** in smart grids.  
The system enables buildings, aggregators, and the grid to act as intelligent agents that coordinate demand response (DR) efficiently while ensuring user comfort, cost savings, and grid stability.

## Problem Statement
Traditional demand response (DR) systems often struggle with:
- Real-time demand variability
- Renewable energy fluctuations
- Diverse consumer behavior  

This project introduces a **MARL framework** where:
- Buildings (consumer agents),  
- Aggregators (regional controllers), and  
- A Grid Controller  

collaborate to optimize power consumption using **hybrid price- and incentive-based signals**.


## Objectives
- Develop a **three-layer MARL architecture** (Consumer, Aggregator, Grid).  
- Implement **hybrid DR mechanisms** (price-based + incentive-based).  
- Train agents with reward functions considering **cost, comfort, and grid stability**.  
- Simulate using the **CityLearn dataset**.  
- Evaluate system performance on:
  - Peak load reduction  
  - Cost efficiency  
  - Fairness and robustness  


## Methodology
- **Algorithms:** Deep Q-Network (DQN), Deep Deterministic Policy Gradient (DDPG).  
- **Benchmarking:** Compared against traditional single-agent and MARL-based DR systems.  
- **Evaluation Metrics:** Peak load reduction, demand fairness, computational efficiency.  

## Expected Outcomes
- Adaptive and decentralized demand response.  
- Improved **grid stability** under peak demand conditions.  
- Balance between **consumer comfort** and **cost savings**.  
- Robustness to edge cases (non-participation, false reporting, overloads).  


## Tech Stack
- **Python 3.x**
- **TensorFlow / PyTorch**
- **OpenAI Gym / CityLearn Environment**
- **NumPy, Pandas, Matplotlib**



## Installation

### 1. Clone Repository
```bash
git clone https://github.com/msaakaash/marl-smartgrid-pricing.git
cd marl-smartgrid-pricing
```

### 2. Create Virtual Environment (recommended)
```bash
python -m venv venv
source venv/bin/activate     
venv\Scripts\activate       
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```txt
numpy==1.26.4
torch==2.2.0
gymnasium==0.29.1
pettingzoo==1.24.3
citylearn==2.3.1
tensorboard==2.16.2
```

## Usage

### 1. Run a Smoke Test
Check if the wrapper runs correctly:
```bash
python wrapper_citylearn_marl.py
```

### 2. Train Agents (Independent A2C)
```bash
python train_marl_a2c.py
```
- Rewards per agent are printed per update.  
- Logs are saved to `runs/` for TensorBoard.

View logs:
```bash
tensorboard --logdir runs
```

### 3. Evaluate Baseline
Run a no-op baseline policy to collect metrics:
```bash
python evaluate.py
```

## Code of Conduct
Please read our [Code of Conduct](docs/CODE_OF_CONDUCT.md) before contributing to this project.

## License  
This project is licensed under the [MIT LICENSE](LICENSE).



## Contributors
- Aakaash M S
- Karthik Ram S  
- Abishek K  
- Ashwin T  


