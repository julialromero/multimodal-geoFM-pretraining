# Hyperbolic embedding visualization

The `hyperbolic_visualization.py` script renders a set of diagnostic plots that
mirror the hyperbolic computations in `ciip.loss`. Run it from the repository
root so that the relative imports resolve correctly.

## Prerequisites

1. A trained CIIP checkpoint (`.pt` file) containing the model weights.
2. The exact training configuration INI that defines the dataset root, band
   selections, and model hyperparameters used for that checkpoint.
3. Access to the SSL4EO dataset directory referenced by the configuration file.

## Basic usage

```bash
python visualization/ssl4eo/hyperbolic_visualization.py \
    --checkpoint /path/to/checkpoint.pt \
    --config ciip/open_clip_train/config_train.ini \
    --output-dir outputs/hyperbolic_viz
```

The command will load the dataset specified in the config, sample every available
season for each selected location, reproduce the hyperbolic loss calculations,
and write the plots plus a `hyperbolic_summary.csv` metadata file to the output
folder.

## Common options

* `--num-locations N` – limit the number of unique location IDs to sample. The
  default is to visit every location in the dataset split referenced by the
  config file.
* `--batch-size B` – adjust the evaluation batch size (defaults to `16`).
* `--device cuda:0` – pick the device on which to run the model forward pass.
* `--no-hyperbolic-normalize` – disable the pre-normalisation step before
  lifting features to the hyperboloid if you want to compare behaviours.

Run `python visualization/ssl4eo/hyperbolic_visualization.py --help` to see the
full set of CLI arguments, including loss-specific overrides like curvature and
margin weight.
