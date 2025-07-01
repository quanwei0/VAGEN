# Multi-Turn RL for LLM Agents

We build our codebase upon [VAGEN](https://github.com/RAGEN-AI/VAGEN).

## Installation

```bash
conda create -n vagen python=3.10 -y
conda activate vagen

bash scripts/install.sh
```


Install Conda (if needed)
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ragen
```


## Usage

```bash
# tmux 1
bash scripts/examples/finegrained/sokoban/grounding_worldmodeling/run_server.sh

# tmux 2
bash scripts/examples/finegrained/sokoban/grounding_worldmodeling/run_train_bilevel_gae.sh
```

wandb read
```bash
wandb sync --view --verbose run-r5iavblg.wandb >> wandb_sync_output.txt
```