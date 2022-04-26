## Readme

### How to train hi-MADRL

To train hi-MADRL with default hyperparameters in both datasets, use:

```
python main_PPO_vecenv.py --dataset purdue --use_eoi --use_hcopo
```

```
python main_PPO_vecenv.py --dataset NCSU --use_eoi --use_hcopo
```

### Outputs

By default, outputs are saved in the directory `../runs/debug`.

### How to test hi-MADRL

```
python main_PPO_vecenv.py --test --output_dir <OUTPUT_DIR>
```

where `<OUTPUT_DIR>` refers to the specific output directories with sub-directories `model`.

### Visualized trajectories

Coming soon...
