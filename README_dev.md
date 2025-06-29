# Multi-Turn RL for LLM Agents

We build our codebase upon [VAGEN](https://github.com/RAGEN-AI/VAGEN).

## Installation

```bash
conda create -n vagen python=3.10 -y
conda activate vagen

bash scripts/install.sh
```

## Usage

```bash
# tmux 1
bash scripts/examples/finegrained/sokoban/grounding_worldmodeling/run_server.sh

# tmux 2
bash scripts/examples/finegrained/sokoban/grounding_worldmodeling/run_train_bilevel_gae.sh
```