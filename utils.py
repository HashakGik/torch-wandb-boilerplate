"""
"THE BEER-WARE LICENSE" (Revision 42):
<https://github.com/HashakGik> wrote this file.  As long as you retain this notice you can do whatever you want
with this stuff. If we meet some day, and you think this stuff is worth it, you can buy me a beer in return.
"""

import random
import time
import multiprocessing
import queue
import numpy as np
import torch
import hashlib
import argparse

import os

from mnemonics import __mnemonic_names__

# Put various utilities in this file. We usually control hyper-parameters and reproducibility from here.
# generate_readme_stub.py expects to find here the classes ArgNumber and ArgBoolean, and the function get_arg_parser().

class ArgNumber:
    """
    Simple numeric argument validator.
    """
    def __init__(self, number_type: type(int) | type(float),
                 min_val: int | float | None = None,
                 max_val: int | float | None = None):
        self.__number_type = number_type
        self.__min = min_val
        self.__max = max_val
        if number_type not in [int, float]:
            raise argparse.ArgumentTypeError("Invalid number type (it must be int or float)")
        if not ((self.__min is None and self.__max is None) or
                (self.__max is None) or (self.__min is None) or (self.__min is not None and self.__min < self.__max)):
            raise argparse.ArgumentTypeError("Invalid range")

    def __call__(self, value: int | float | str) -> int | float:
        try:
            val = self.__number_type(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid value specified, conversion issues! Provided: {value}")
        if self.__min is not None and val < self.__min:
            raise argparse.ArgumentTypeError(f"Invalid value specified, it must be >= {self.__min}")
        if self.__max is not None and val > self.__max:
            raise argparse.ArgumentTypeError(f"Invalid value specified, it must be <= {self.__max}")
        return val


class ArgBoolean:
    """
    Simple Boolean argument validator.
    """
    def __call__(self, value: str | bool | int) -> bool:
        if isinstance(value, str):
            val = value.lower().strip()
            if val != "true" and val != "false" and val != "yes" and val != "no":
                raise argparse.ArgumentTypeError(f"Invalid value specified: {value}")
            val = True if (val == "true" or val == "yes") else False
        elif isinstance(value, int):
            if value != 0 and value != 1:
                raise argparse.ArgumentTypeError(f"Invalid value specified: {value}")
            val = value == 1
        elif isinstance(value, bool):
            val = value
        else:
            raise argparse.ArgumentTypeError(f"Invalid value specified (expected boolean): {value}")
        return val

class Watchdog:
    """
    Watchdog timer for a training loop. It runs training on a separate process, monitoring it with a periodic heartbeat.
    Optionally, if a timeout is set, ensures that each epoch runs at most for that duration (approximated to the next integer multiple of the heartbeat).
    There are two reasons why this is required:
    1) signal.alarm only works on UNIX systems
    2) we usually run portions of native code (C++/Rust), over which we have no control, this means that any exception raised
       during native code execution will be lost. If exceptions are handled by a separate process, however, we can catch them
       (at the expenses of killing the subprocess, without graceful termination).

    If neither of these conditions apply to your setting (i.e., a pure torch training loop on UNIX), simply wrap your training code
    with a signal.alarm raising a TimeoutException.
    """
    def __init__(self, function, heartbeat, *args, **kwargs):
        """
        Initializes an internal message queue and the child process.
        :param function: Function run by the child process. It must be a generator yielding triples (phase, values, tags).
                         The phase is one of ["start", "epoch", "end"].
        :param heartbeat: Heartbeat duration (in seconds).
        :param args: Positional arguments passed to the function.
        :param kwargs: Keyword arguments passed to the function.
        """
        self.queue = multiprocessing.Queue()
        self.heartbeat = heartbeat

        self.fn_proc = multiprocessing.Process(target=self._fn_wrapper(function), args=args, kwargs=kwargs,
                                               daemon=True)

    def _fn_wrapper(self, function):
        """
        Simple generator wrapper. It writes yielded values to the message queue.
        :param function: Generator function.
        :return: The wrapped function.
        """
        def f(*args, **kwargs):
            for x in function(*args, **kwargs):
                self.queue.put(x)

        return f

    def listen(self, timeout):
        """
        Listen on the message queue for messages (phase, values, tags) written by the subprocess.
        It relies on two sentinels: phase="start" and phase="end".
        :param timeout: Timeout (in seconds) after which it gives up waiting for a new message when phase="epoch".
                        It is rounded to the next integer multiple of self.heartbeat.
        :return: The tuple (full training history, training tags).
        """
        finished = False
        epoch_time = 0
        history = []  # Keep a local copy of the training history updated incrementally, to have some data to report in case of crashing/timeouts.
        return_tags = set()
        _, _, _ = self.queue.get(block=True) # Always block until the first sentinel is received.
        while not finished:
            if not self.fn_proc.is_alive():
                return_tags.add("Crashed")
                break

            try:
                phase, epoch_stats, tags = self.queue.get(block=True, timeout=self.heartbeat)
                epoch_time = 0 # Reset epoch timeout.

                if phase == "epoch":
                    history.append(epoch_stats)  # Update partial history.
                else:  # Reached the end of the training loop (received the sentinel phase = "end").
                    finished = True
                    history = epoch_stats  # Replace the entire history at the end of training, since it completed successfully.

                return_tags.update(tags)

            except queue.Empty:
                pass  # In case of heartbeat timeout, there is still a chance of being within the epoch timeout limit.
            except KeyboardInterrupt:
                return_tags.add("User abort")
                break

            epoch_time += self.heartbeat

            if timeout > 0 and epoch_time >= timeout:
                return_tags.add("Timeout")  # If the epoch timeout has expired, give up.
                break

        return history, return_tags

    def __enter__(self):
        self.fn_proc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fn_proc.is_alive():
            self.fn_proc.kill()
            self.fn_proc.join()
            self.fn_proc.close()

        return False # In case some unhandled exception is called, let them propagate to the caller.

def get_arg_parser():
    """
    Build the argument parser. Define here all the experiment hyper-parameters and house-keeping arguments
    (e.g., whether to save weights at the end of training). For readability, we usually group them by category.
    To reduce the (already large) number of arguments, we try to group related HPs with a colon-separated string
    (e.g., "mlp:NUM_LAYERS:ACTIVATION") and we split it during parsing. This is also beneficial to reduce grid searches.

    generate_readme_stub.py will parse the help field automatically. Default values should be put at the end of the string,
    as "(default: value)", which will be deleted, or as "(comment; default: value)", which will become "(comment)".
    :return: An argparse.ArgumentParser
    """
    arg_parser = argparse.ArgumentParser()
    # Dataset parameters.
    arg_parser.add_argument('--prefix_path', help="Path to the root of the project (default: '.')", type=str,
                            default=".")
    arg_parser.add_argument('--data_path',
                            help="Path to the data folder, relative to root (default: 'data')",
                            type=str, default="data")
    arg_parser.add_argument('--output_path',
                            help="Path to the output folder, relative to root (default: 'outputs')",
                            type=str, default="outputs")

    # Training parameters.
    arg_parser.add_argument("--lr", help="Learning rate (positive: SGD, negative: Adam; default: -1e-3)",
                            type=ArgNumber(float), default=-1e-3)
    arg_parser.add_argument("--epochs", help="Training epochs (default: 10)", type=ArgNumber(int, min_val=1),
                            default=10)
    arg_parser.add_argument("--batch_size", help="Batch size (default: 128)", type=ArgNumber(int, min_val=1), default=128)
    arg_parser.add_argument("--supervision_lambda", help="Weight for direct supervision (default: 1.0)",
                            type=ArgNumber(float, min_val=0.0), default=1.0)
    arg_parser.add_argument('--grad_clipping', help="Norm for gradient clipping, disable if 0.0 (default: 0.0)",
                            type=ArgNumber(float, min_val=0.0), default=0.0)

    # Model parameters.
    # [...]

    # Experiment parameters.
    arg_parser.add_argument('--seed',
                            help="Integer seed for random generator (if negative, use timestamp; default: -1)",
                            type=int, default=-1)
    arg_parser.add_argument('--epoch_timeout',
                            help="Timeout for each epoch, in minutes (disable if 0; default: 0)",
                            type=ArgNumber(int, min_val=0), default=0)
    arg_parser.add_argument('--device',
                            help="Device to use, no effect if environment variable DEVICE is set (default: 'cpu')",
                            type=str, default="cpu")
    arg_parser.add_argument('--save', help="Save network weights and results locally (default: False)",
                            type=ArgBoolean(), default=False)
    arg_parser.add_argument('--abort_irrelevant',
                            help="Abort irrelevant combinations of hyper-parameters (useful when sweeping grid searches; default: True)",
                            type=ArgBoolean(), default=True)
    arg_parser.add_argument("--wandb_project", help="Use W&B, this is the project name (default: None)", type=str,
                            default=None)
    arg_parser.add_argument('--verbose',
                            help="Amount of output (0: no output, 1: epoch summary, 2: full; default: 1)",
                            type=int, choices=[0, 1, 2], default=1)

    # Magic numbers.
    arg_parser.add_argument('--eps',
                            help="Epsilon for numerical stability (default: 1e-7)",
                            type=ArgNumber(float, min_val=0), default=1e-7)
    arg_parser.add_argument('--overfitting_threshold',
                            help="Threshold triggering an overfitting tag (default: 0.5)",
                            type=ArgNumber(float, min_val=0), default=0.5)
    arg_parser.add_argument('--vanishing_threshold',
                            help="Threshold triggering a vanishing gradient tag (default: 1e-5)",
                            type=ArgNumber(float, min_val=0), default=1e-5)
    arg_parser.add_argument('--exploding_threshold',
                            help="Threshold triggering an exploding gradient tag (default: 1e7)",
                            type=ArgNumber(float, min_val=0), default=1e7)
    arg_parser.add_argument('--std_threshold',
                            help="Threshold triggering an high gradient standard deviation tag (default: 200.0)",
                            type=ArgNumber(float, min_val=0), default=200.0)
    arg_parser.add_argument('--rnd_threshold',
                            help="Threshold triggering a random guessing tag (default: 0.1)",
                            type=ArgNumber(float, min_val=0), default=0.1)
    arg_parser.add_argument('--mp_threshold',
                            help="Threshold triggering a most probable guessing tag (default: 0.1)",
                            type=ArgNumber(float, min_val=0), default=0.1)

    return arg_parser

def get_unhashable_opts():
    """
    Explicit list of arguments which should NOT be used for the unique identifier computation.
    This list should contain at least the random seed (because we want to group together different randomized runs),
    and every house-keeping argument which does not affect the reproducibility of a single run (to allow the same
    run computed on different machines to be hashed in the same way).
    :return: The list of excluded arguments.
    """
    return ["wandb_project", "save", "seed", "device", "prefix_path", "data_path", "output_path", "verbose",
            "abort_irrelevant", "eps", "overfitting_threshold", "vanishing_threshold", "exploding_threshold",
            "std_threshold", "rnd_threshold", "mp_threshold", "epoch_timeout"] # epoch_timeout may be an important hyper-parameter, depending on the type of experiment.

def generate_name(opts, mnemonics=None):
    """
    Get a deterministic identifier for a group of experiments, based on the MD5 hash of relevant hyper-parameters.
    To avoid collisions, the identifier is a mnemonic name, followed by the first 4 bytes of the salted and re-hashed hyper-parameters.
    Important: float hyper-parameters are truncated at the 6th decimal place, therefore, e.g., lr=1e-7 and lr=1e-8 *WILL* collide:
    do not use this function with a random search on a non-discrete space.
    Note: The 'dogs' namespace is significantly larger than others to reduce collisions in hyper-parameter-rich experiments.
    :param opts: Dictionary of hyper-parameters. The identifier will be computed based on its content.
    :param mnemonics: Which namespace to use.
    It can be a string in {'cats', 'dogs', 'cows', 'ohio', 'nations', 'german_expressionism', 'bava', 'kurosawa', 'scorsese'},
    a custom list of strings, or None (in which case, the namespace will be chosen based on the W&B project name).
    :return: A unique string identifier for the current experiment.
    """

    groupable_opts = set(opts.keys()).difference(get_unhashable_opts())

    if mnemonics is None:
        if "wandb_project" not in opts or opts["wandb_project"] is None:
            mnemonics = "cows"
        else:
            project_hash = int(hashlib.md5(opts["wandb_project"].encode("utf-8")).hexdigest(), 16)
            mnemonics = sorted(__mnemonic_names__.keys())[project_hash % len(__mnemonic_names__)]

    unique_id = ""
    for o in sorted(groupable_opts):
        if isinstance(opts[o], float):
            unique_id += "{:.6f}".format(opts[o])
        else:
            unique_id += str(opts[o])
    hash = hashlib.md5(unique_id.encode("utf-8")).hexdigest()

    assert isinstance(mnemonics, str) or isinstance(mnemonics, list), \
        "Mnemonics must be a string or a list of strings. Found {}.".format(type(mnemonics))
    if isinstance(mnemonics, str):
        assert mnemonics in __mnemonic_names__.keys(), \
            "Unknown mnemonics group. It must be one of {{{}}}".format(", ".join(__mnemonic_names__.keys()))
        # Salting is required to minimize the risk of collisions on small lists of mnemonic names.
        salted_hash = hashlib.md5("{}{}".format(unique_id, mnemonics).encode("utf-8")).hexdigest()
        idx = int(hash, 16) % len(__mnemonic_names__[mnemonics])
        out = "{}_{}".format(__mnemonic_names__[mnemonics][idx], salted_hash[:len(salted_hash) // 4])
    else:
        salted_hash = hashlib.md5("{}{}".format(unique_id, mnemonics[0]).encode("utf-8")).hexdigest()
        idx = int(hash, 16) % len(mnemonics)
        out = "{}_{}".format(mnemonics[idx], salted_hash[:len(salted_hash) // 4])

    return out


def set_seed(seed):
    """
    Set the seed of random number generators in a paranoid manner.
    Note that some code may still be non-deterministic, especially in imported libraries.
    :param seed: Random seed, if < 0, it will use the current timestamp.
    :return: A seeded numpy.random.Generator.
    """
    seed = int(time.time()) if seed < 0 else int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return np.random.default_rng(seed)

def preflight_checks(opts):
    """
    Perform important checks before starting an experiment. This function should fail for experiment-breaking errors
    (e.g., missing dataset folders, invalid values of hyper-parameters, etc.).
    Optionally modifies the hyper-parameters dictionary to convert some options from human-friendly to machine-friendly.
    :param opts: Dictionary of hyper-parameters.
    :return: The modified dictionary of HPs.
    """

    # Sanity checks.
#    assert os.path.exists("{}/{}".format(opts["prefix_path"], opts["data_path"])), \
#    "Dataset not found at {}/{}.".format(opts["prefix_path"], opts["data_path"])

    # Modify hyper-parameters and perform further checks.
    #tmp = opts["model"].split(":")
    #opts["model"] = {"type": tmp[0], "num_layers": tmp[1], "activation": tmp[2]}
    # assert opts["model"]["type"] in ["mlp", "lstm", "bert", "skynet"]
    # assert opts["model"]["type"] != "bert" or opts["model"]["num_layers"] <= 6, \
    # "Do not use a too deep architecture (layers > 6) with BERT"

    return opts

def prune_hyperparameters(opts, arg_parser):
    """
    Check whether irrelevant hyper-parameters have a non-default value. This allows to prune grid search space by
    skipping combinations of already-tried experiments.
    :param opts: Dictionary of hyper-parameters.
    :param arg_parser: argparse.ArgumentParser.
    :return: True if this experiment can be performed, False if it should be skipped.
    """

    ok = True

    loss_weights = sum([v for k, v in opts.items() if k.endswith("_lambda")])
    if loss_weights <= 0:
        ok = False
        print("Warning: At least one loss function must be enabled.")

    #if opts["model"]["type"] == "lstm" and opts["backprop_through_time"] != arg_parser.get_default("backprop_through_time"):
    #    ok = False
    #    print("Warning: Backpropagation through time makes sense only when training recurrent neural networks.")

    return ok