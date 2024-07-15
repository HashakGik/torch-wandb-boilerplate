"""
"THE BEER-WARE LICENSE" (Revision 42):
<https://github.com/HashakGik> wrote this file.  As long as you retain this notice you can do whatever you want
with this stuff. If we meet some day, and you think this stuff is worth it, you can buy me a beer in return.
"""

import torch
from torch.utils.data import DataLoader

import tqdm.auto as tqdm
import signal # Note: this will not work for non-UNIX systems, as we rely on SIGALRM to detect timeouts.
import time

import torcheval.metrics
from metrics import Variance, grad_norm

# Use this file for your training logic. We usually need highly-customized training loops, so a standardized interface
# like Lightning does not suit our needs.


def train(net, train_ds, val_ds, test_ds, rng, opts):
    """
    Train the model on the provided datasets. If some events are triggered during training, it collects them on a set of flags.
    These events can be errors (NaN loss, infinite gradients, etc.), warnings (e.g., vanishing gradients, overfitting),
    interrupts (e.g., timeout, user abort, etc.), or informative (e.g., successful run).
    Depending on the experimental setting, some may be more important than others.
    :param net: torch.nn.Module to train.
    :param train_ds: Train dataset.
    :param val_ds: Validation dataset.
    :param test_ds: Test dataset.
    :param opts: Dictionary of hyper-parameters.
    :param rng: Seeded numpy.random.Generator.
    :return: Tuple: (history: list of metrics evaluated for each epoch, return_flags: set of events triggered during training).
    """
    def timeout_handler(signum, frame):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, timeout_handler) # NOTE: this works only on UNIX systems!

    train_dl = DataLoader(train_ds, batch_size=opts["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=opts["batch_size"], shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=opts["batch_size"], shuffle=False)

    net.to(opts["device"])

    if opts["lr"] > 0.0:
        optimizer = torch.optim.SGD(net.parameters(), lr=opts["lr"])
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=-opts["lr"])

    if opts["grad_clipping"] > 0.0:
        for p in net.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -opts["grad_clipping"], opts["grad_clipping"]))

    history = []
    return_flags = set()
    ok = True # True for successful completion and completion with warnings, False for errors.

    # Define here your metrics.
    metrics = {
        s: {
            "accuracy": torcheval.metrics.MulticlassAccuracy(device=opts["device"]),
            "f1": torcheval.metrics.MulticlassF1Score(device=opts["device"]),
        } for s in ["train", "val", "test"]
    }


    # Additional metrics.
    metrics["train"]["loss"] = torcheval.metrics.Mean(device=opts["device"])
    metrics["train"]["avg_grad_norm"] = torcheval.metrics.Mean(device=opts["device"])
    metrics["train"]["var_grad_norm"] = Variance(device=opts["device"])


    for e in tqdm.trange(opts["epochs"], position=0, desc="epoch"):
        for v in metrics.values():
            for v2 in v.values():
                v2.reset()

        net.train()

        # Set epoch timeout in minutes. This time includes both training and evaluation stages.
        if opts["epoch_timeout"] > 0:
            signal.alarm(opts["epoch_timeout"] * 60)

        try:
            start_time = time.time()
            # Train the model.
            for batch in (bar := tqdm.tqdm(train_dl, position=1, desc="batch", leave=False, ncols=0)):
                loss, ok, step_flags = train_step(net, optimizer, batch, metrics, opts)

                return_flags.update(step_flags)

                bar.set_postfix_str(
                    "Loss: {:.02f}, avg_Loss: {:.02f}, dLoss: {:.02f}, dLoss_var: {:.02f}, Accuracy: {:.02f}, F1: {:.02f}".format(
                        loss, metrics["train"]["loss"].compute(), metrics["train"]["avg_grad_norm"].compute(),
                        metrics["train"]["var_grad_norm"].compute(), metrics["train"]["accuracy"].compute(),
                        metrics["train"]["f1"].compute()
                    )
                )
                if not ok:
                    break

            times = {"train": time.time() - start_time}
            with torch.no_grad():
                net.eval()
                # Evaluate the model. If the validation set needs to be used to perform some calibration, split this loop into two.
                for split, dl in {"val": val_dl, "test": test_dl}.items():
                    start_time = time.time()
                    for batch in (bar := tqdm.tqdm(dl, position=1, desc="batch", leave=False, ncols=0)):
                        eval_step(net, batch, metrics, split, opts)
                    times[split] = time.time() - start_time

                    bar.set_postfix_str(
                        "({}) Accuracy: {:.02f}, F1: {:.02f}".format(
                            split,
                            metrics[split]["accuracy"].compute(),
                            metrics[split]["f1"].compute()
                        )
                    )

            epoch_stats = {}
            for split in ["train", "val", "test"]:
                epoch_stats["{}_time".format(split)] = times[split]
                for k, v in metrics[split].items():
                    epoch_stats["{}_{}".format(split, k)] = v.compute().item()

            # "Generalization" metrics: train-test and train/test for every recorded metric.
            for k in metrics["test"].keys():
                epoch_stats["delta_{}".format(k)] = epoch_stats["train_{}".format(k)] - epoch_stats["test_{}".format(k)]
                if epoch_stats["delta_{}".format(k)] > 0.5:
                    return_flags.add("Overfitting {} (learning)".format(k))

                if epoch_stats["test_{}".format(k)] != 0:
                    epoch_stats["ratio_{}".format(k)] = epoch_stats["train_{}".format(k)] / epoch_stats[
                        "test_{}".format(k)]
                else:
                    epoch_stats["ratio_{}".format(k)] = 0.0

            history.append(epoch_stats)

        # If the timer has expired, abort training.
        except TimeoutError:
            return_flags.add("Timeout")
            ok = False

        # If the user presses Ctrl+C, abort training. It does not work when W&B is running in sweep mode, because
        # the wrapper will catch the signal before this exception.
        except KeyboardInterrupt:
            return_flags.add("User abort")
            ok = False

        # Whatever happens, disable the timer at the end.
        finally:
            signal.alarm(0)

        if not ok:
            break

    if ok:
        for k in metrics["test"].keys():
            if history[-1]["delta_{}".format(k)] > 0.5:
                return_flags.add("Overfitting {} (end)".format(k))

        return_flags.add("Success")

    return history, return_flags

def train_step(net, optimizer, batch, metrics, opts):
    """
    Single training step.
    :param net: torch.nn.Module to train.
    :param optimizer: torch.optim optimizer.
    :param batch: Input batch.
    :param metrics: Metrics to update at the end of the step.
    :param opts: Dictionary of hyper-parameters.
    :return: Tuple (loss: value of the loss, ok: whether training should continue, ret_flags: events triggered by this batch)
    """
    ok = True
    ret_flags = set()
    img, label = batch

    img = img.to(opts["device"])
    label = label.to(opts["device"])

    pred = net(img)

    loss = opts["supervision_lambda"] * torch.nn.functional.cross_entropy(pred, label)
    # loss += opts["other_lambda"] * other_loss()

    if torch.isnan(loss):
        ok = False
        ret_flags.add("NaN loss")
    elif torch.isposinf(loss):
        ok = False
        ret_flags.add("Inf loss")
    else:
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    metrics["train"]["loss"].update(loss)
    norms = grad_norm(list(net.parameters()))

    if torch.isnan(norms):
        ok = False
        ret_flags.add("NaN gradient")

    # These magic numbers depend on the experiment.
    if norms < 1e-5:
        print("Warning: Vanishing gradients. Gradient norm: {}".format(norms))
        ret_flags.add("Vanishing gradient")
    elif norms > 1e10:
        print("Warning: Exploding gradients. Gradient norm: {}".format(norms))
        ret_flags.add("Exploding gradient")

    metrics["train"]["avg_grad_norm"].update(norms)
    metrics["train"]["var_grad_norm"].update(norms)

    if metrics["train"]["var_grad_norm"].compute() > 1e7:
        print("Warning: High gradient variance.")
        ret_flags.add("High variance")

    # Update metrics.
    metrics["train"]["accuracy"].update(pred, label)
    metrics["train"]["f1"].update(pred, label)

    return loss, ok, ret_flags

def eval_step(net, batch, metrics, split, opts):
    """
    Single evaluation step.
    :param net: torch.nn.Module to evaluate.
    :param batch: Input batch.
    :param metrics: Metrics to update.
    :param split: Dataset split.
    :param opts: Dictionary of hyper-parameters.
    """
    img, label = batch

    img = img.to(opts["device"])
    label = label.to(opts["device"])

    pred = net(img)

    metrics[split]["accuracy"].update(pred, label)
    metrics[split]["f1"].update(pred, label)