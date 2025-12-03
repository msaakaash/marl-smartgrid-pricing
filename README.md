<h1 align="center">MARL Smart Grid Pricing</h1>

## Overview
 
This project introduces a **Multi-Agent Reinforcement Learning (MARL)** framework for smart grid pricing, where **Consumers**, **Aggregators**, and a **Grid Controller** collaborate to optimize energy consumption using **hybrid price and incentive-based mechanisms**.  

## Features
- **Three-layer MARL architecture** with Consumer, Aggregator, and Grid agents.  
- **Hybrid demand response** combining price-based and incentive-based strategies.  
- **Reinforcement learning algorithms**: DQN and DDPG.  
- **Reward optimization** for cost, comfort, and grid stability.  
- **Performance evaluation** on peak load reduction, cost efficiency, and fairness.  
- **Scalable and adaptive** system robust to non-participation and edge conditions.
- **Security** with  ChaCha20‑Poly1305 - a modern, fast, and authenticated encryption scheme.

## Tech Stack

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Tools / Technologies</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Programming Language</td>
      <td>Python 3.10</td>
    </tr>
    <tr>
      <td>Frameworks / Libraries</td>
      <td>
        PyTorch – Deep learning and reinforcement learning models<br>
        CityLearn – Urban building energy simulation<br>
        NumPy, Pandas – Numerical computation and data processing<br>
        scikit-learn – Data preprocessing and utilities<br>
        Gym – Reinforcement learning environment interface<br>
        tqdm – Progress visualization<br>
        cryptography - Security 
      </td>
    </tr>
    <tr>
      <td>Tools & Environment</td>
      <td>
        Virtual Environment (venv)<br>
        Git & GitHub for version control<br>
        Jupyter Notebook (optional, for analysis & visualization)
      </td>
    </tr>
  </tbody>
</table>

## Installation Guide

> Use **Python 3.10** for best compatibility.


#### 1) Clone the Repository

```bash
git clone https://github.com/yourusername/marl-smartgrid-pricing.git
cd marl-smartgrid-pricing
```

#### 2) Create Python 3.10 Virtual Environment

**Windows (PowerShell)**

```bash
python -m venv marl_env310
marl_env310\Scripts\activate
```

**macOS/Linux**

```bash
python3 -m venv marl_env310
source marl_env310/bin/activate
```

#### 3) Install Dependencies

> **Note:**   
> - After activating the virtual environment, upgrade pip before installing dependencies:  
>   ```bash
>   pip install --upgrade pip
>   ```
> - Install all the requirements:
>   ```bash
>   pip install -r requirements.txt
>   ```



If you don’t have a `requirements.txt` file, create it with the following content:

```text
citylearn==1.5.0
numpy==1.21.6
pandas==1.3.5
torch==1.12.0
scikit-learn==1.0.2
gym==0.25.1
tqdm==4.66.1
cryptography==41.0.3
matplotlib==3.5.3
```

## Usage

Run the main script:

```bash
python main.py
```




## Code of Conduct
Please read our [Code of Conduct](docs/CODE_OF_CONDUCT.md) before contributing to this project.

## Contributing
Contributions are welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.


## Security
If you discover a vulnerability, please refer to our [Security Policy](docs/SECURITY.md) for instructions on how to report it responsibly.


## License  
This project is licensed under the [MIT LICENSE](LICENSE).



## Contributors
- Aakaash M S
- S Karthik Ram
- Abishek K  
- Ashwin T  


