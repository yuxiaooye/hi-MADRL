## Readme

### How to train hi-MADRL

To train hi-MADRL with default hyperparameters in both datasets, use:

```
python main_PPO_vecenv.py --dataset purdue --use_eoi --use_hcopo
```

```
python main_PPO_vecenv.py --dataset NCSU --use_eoi --use_hcopo
```

By default, the experiment are conducted under the simulation settings summarized in Table 2.

### Outputs

By default, outputs are saved in the directory `../runs/debug`.

`train_output.txt` records the performance in terms of 5 metrics:

```
best trajs have been changed in ts=200. best_train_reward: 0.238 efficiency: 2.029 collect_data_ratio: 0.550 loss_ratio: 0.011 fairness: 0.577 uav_util_factor: 1.714 energy_consumption_ratio: 0.155 
```

### How to test hi-MADRL

```
python main_PPO_vecenv.py --test --output_dir <OUTPUT_DIR>
```

where `<OUTPUT_DIR>` refers to the specific output directories with sub-directories `model`.

### Visualized trajectories

Coming soon...
