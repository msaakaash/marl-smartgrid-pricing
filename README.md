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


## 🛠️ Tech Stack
- **Python 3.x**
- **TensorFlow / PyTorch**
- **OpenAI Gym / CityLearn Environment**
- **NumPy, Pandas, Matplotlib**

## Contributors
- M S Aakaash  
- Karthik Ram S  
- Abishek K  
- Ashwin T  

Guide: **Dr. Radhika N**
