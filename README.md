# Comparison of FNO and SFNO for solving the Shallow Water Equations
this is mainly made to demonstrate how you can combine the following tools
1. hydra
2. lightning
3. wandb
and use them efficiently across different hardware setups, e.g. your laptop for debugging and a SLURM cluster for training. You'll find explanations on it on my [blog](https://liopeer.github.io/mlblog/posts/2024-06-18_hydra_wandb/)
## Launching Training
```bash
python train.py --config-name=neuraloperator
```
Since the CLI uses Hydra, you can override any hyperparams in the config (also see the blog for that).