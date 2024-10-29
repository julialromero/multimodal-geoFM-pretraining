# General Notes
This repository can operate inside either a conda or python virtual environment.  I've had trouble stabilizing the conda environment, so right now, using python venv and installing modules with pip is likely your best bet.

We're using hydra to run/iterate/output.  It automatically generates files in the 'outputs' folder, delineating the day and time of the run.  Inside those folders, you'll see the configurations for the run, outputs, logs, etc.  Hydra is pretty useful; it essentially uses a default configuration file, and then allows you to mix and match any number of overrides for test runs.  In our case, our configs are things like learning rate, batch size, bands, etc.  There's some excellent documentation on it here: https://hydra.cc/docs/intro/

# Environment Setup
1. This is using a python3 venv setup; your command should look something like this: python3 -m venv .ciip_env
2. I'm using Python 3.9.6; you can specify python version in making your virtual env as needed
3. Activate the environment by running source {your_env_name}/bin/activate
4. With the environment active, ensure pip is up to date with pip install --upgrade pip
5. From there, run pip install -r requirements.txt to import necessary modules
6. Ensure that the interpreter for your project in your IDE (whether VSCode or Pycharm) is set to the python version from your virtual environment

# Local Testing
This will let you have some quick sanity checking for syntax/environment issues before moving files onto Hal or Alpine for more rigorous iteration.  If you want to adjust any configurations for this, you can do so in the configs/local_default.yaml file.
1. Go to https://drive.google.com/file/d/1sRWcYbaWs-efXza6kw03GlJQdZHq5iRN/view?pli=1, download the data, then move the folders labelled s1 and s2c into a new directory at the same level as ciip called "local_test_data".
2. We need to ensure that hydra is pointed at a config file for small, local runs.  In your run_train_val class, change line 21 to look like this: CONF="local_default"
3. Now, we can do a quick run; you should be running from within the {your_path_to_repo}/ciip/ciip/open_clip_train/ directory: 
   4. In the command line, it'll just look like running python3 run_train_val.py; you can feed in additional hydra configs if you want to (see below)
   6. You can setup your IDE to run this as well by clicking 'edit configurations' for the run_train_val.py class and ensuring that it matches what's above.
4. You should now see some output in {your_path_to_repo}/ciip/ciip/open/clip_train/outputs/YYYY-MM-DD/hh-mm-ss/ folder
   5. Hydra generates a new folder per run according to that filepath setup
   6. Inside .hydra, it'll denote the configurations for the run and any overrides that were present (see below)
   6. It'll pull in any logs that you generate via python's default logging class; in this case, you'll see run_train_val.log generated
   7. I've also set it up so that checkpoints are routed into this output directory
   8. Any run-specific information that you want to track, you should be able to store 
   9. This folder should be ignored in git; keep your own outputs localized

# Production Testing
This is for when you're iterating through different options.
1. Make sure line 21 of your run_train_val looks like this: CONF="prod_default"
2. You can fire off a single run with the exact same command specified above
3. If you want to start messing around with hyperparameters, you can start to add specific config files that override the default hyperparams found in prod_default.yaml.  I've included some sample overrides; the datamodule and train folders both pertain to the subsections in the prod_default.yaml file.  If I wanted to try out these different permutations, I would run something that looks like this: python3 run_train_val.py hydra.job.chdir=False hydra.mode=MULTIRUN datamodule=large_batch,small_batch train=high_learning_rate.
   4. Hydra will then fire off 2 runs, where it's mixing and matching batch sizes against the high learning rate.

# TODO
- add cometML for logging
  - document necessary online setup for cometML