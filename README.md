
# Boilerplate for Pytorch and Weights and Biases

During our experiments, I often found myself reimplementing the same training loop multiple times.
This leads to duplicate code, the chance to introduce new bugs and propagate old ones, and a slow start for new projects.

This repository is a simple boilerplate which suits our needs for machine learning experiments.
It features a customizable training loop in Pytorch and exploits [Weights & Biases (W&B)](https://wandb.ai) to log results in an organized fashion.

Experimental hyper-parameters are controlled via command-line arguments and can be easily searched with W&B sweeps.
Experiments are automatically grouped by hyper-parameter values, to easily compute averages and errors over multiple replicates.
Events triggered during training are tracked and each run is tagged accordingly, to easily filter and debug problematic runs.

**Caveat:** We often deal with a **discrete** hyper-parameter space (i.e., a grid search or a random search on discrete values), 
and we train from scratch. This boilerplate is optimized for our needs, and it may need to be adapted.

A small snippet also allows to automatically generate a basic `README.md` containing code dependencies and command-line arguments. 

For demonstration purposes, this boilerplate trains a simple feedforward network on the MNIST digits dataset.

You can check the results [here](https://wandb.ai/l-lorello/boilerplate).
## License
For anti-plagiarism reasons, we need to share our experiments with an attribution license, 
however, this boilerplate is released under the *Beer-ware license*.

```
/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * <https://github.com/HashakGik> wrote this file.  As long as you retain this notice you
 * can do whatever you want with this stuff. If we meet some day, and you think
 * this stuff is worth it, you can buy me a beer in return.
 * ----------------------------------------------------------------------------
 */
```

Some design choices in the code were inherited from [my supervisor](https://github.com/mela64).
Generated experiment names are property of their respective owners.

## Requirements

Requirements will depend on the actual implementation, here is a minimum list of dependencies:

```
wandb
torch
torcheval
torchvision
tqdm
```

For the `generate_readme_stub.py` snippet, make sure `pipreqs` is installed.

## Usage (generate *README.md*)
1. Delete this `README.md`
2. Run `python generate_readme_stub.py`
3. Delete `generate_readme_stub.py` (or gitignore it)
4. Manually complete new `README.md`

## Usage (standalone mode)
1. Run `python main.py [--args values] --save yes/no`, use `--help` for a list of hyper-parameters.
2. Analyze results printed on screen (`--save no`) or stored on `--output_path`.

**Note:** you can pass the `--device` argument or set the `DEVICE` environment variable to control on which device the experiment will be run.

## Usage (standalone W&B)
1. Run `python main.py [--args values] --wandb_project "project name"`.
2. Log into your W&B account and analyze results from your dashboard.
3. Group project runs by "Group" and filter them by "Tags".

## Usage (W&B sweep)
1. Define a [W&B Sweep](https://docs.wandb.ai/guides/sweeps) by searching at least over a `seed` hyper-parameter
2. Run your sweep with `wandb agent sweep_id` (also from multiple machines).
3. Group project runs by "Group" and filter them by "Tags".
4. Analyze results from your dashboard.

**Note:** runs will be automatically associated with a Finished/Failed status, based on training events.

**Caveat:** Due to how wandb agent wraps the code, `KeyboardInterrupt` will never be caught, so the `User abort` tag will never be generated in sweep mode.

If your hyper-parameter space is sparse, you can set the following environment variables, to prevent `wandb agent` from stopping:
```
export WANDB_AGENT_DISABLE_FLAPPING=true
export WANDB_AGENT_MAX_INITIAL_FAILURES=1000
```
## List of tags

| Tag  (informative)                           | Event                                                                                                                                                        |
|----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Success`                                    | Training completed successfully                                                                                                                              |
| `Overfitting METRIC_NAME (learning)`         | The difference of `METRIC_NAME` between training and test set is too high at some point during training                                                      |
| `Overfitting METRIC_NAME (end)`              | The difference of `METRIC_NAME` between training and test set is still too high at the end of training                                                       |
| `Random guessing METRIC_NAME (SPLIT)`        | At the end of training, the model performs as bad as (or worse) a random guesser, in terms of `METRIC_NAME` measured on `SPLIT`                              |
| `Most probable guessing METRIC_NAME (SPLIT)` | At the end of training, the model performs as bad as (or worse) a model which guesses the most probable class, in terms of `METRIC_NAME` measured on `SPLIT` |
| `Vanishing gradient`                         | The gradient norm is too small for some batch                                                                                                                |
| `Exploding gradient`                         | The gradient norm is too large for some batch                                                                                                                |
| `High StdDev`                                | The standard deviation of the gradient norm is too large for some batch                                                                                      |

| Tag (error)                                  | Event                                                                                                                                                        |
|----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Irrelevant hyperparameters`                 | Experiment aborted because `--abort_irrelevant` was passed and a combination of hyper-parameters was irrelevant                                              |
| `User abort`                                 | The used sent a `SIGINT` signal                                                                                                                              |
| `Timeout`                                    | `--epoch_timeout` was set and a single epoch took too long to finish                                                                                         |
| `NaN loss`                                   | The loss was not a number                                                                                                                                    |
| `Inf loss`                                   | The loss was infinite                                                                                                                                        |
| `NaN gradient`                               | The gradient norm was not a number                                                                                                                           |



**Irrelevant hyper-parameters:** Sometimes, hyper-parameters interact in complex ways and some combinations may be redundant
(e.g., when comparing an MLP with a LSTM, backpropagation through time makes sense only in the latter case).
W&B has no way of nesting hyper-parameters and it will repeat experiments, potentially sweeping through combinations which
are functionally equivalent. In order to save precious search time, the `--abort_irrelevant` argument terminates a run when
the `utils.prune_hyperparameters` function detects an irrelevant combination.

### Thresholds for tags

| Tag                      | Argument                  | Default value | Trigger condition                           |
|--------------------------|---------------------------|---------------|---------------------------------------------|
| `Overfitting`            | `--overfitting_threshold` | 0.5           | $`M_{train} - M_{test} > threshold`$        |
| `Vanishing gradient`     | `--vanishing_threshold`   | 1e-5          | $`\|\nabla Loss\| < threshold`$             |
| `Exploding gradient`     | `--exploding_threshold`   | 1e7           | $`\|\nabla Loss\| > threshold`              | 
| `High StdDev`            | `--std_threshold`         | 200.0         | $`\sigma(\nabla Loss) > threshold`$         |
| `Random guessing`        | `--rnd_threshold`         | 0.1           | $`M^{rnd}_{split} + threshold > M_{split}`$ |
| `Most probable guessing` | `--mp_threshold`          | 0.1           | $`M^{mp}_{split} + threshold > M_{split}`$  |