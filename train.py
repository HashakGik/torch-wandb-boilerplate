"""
"THE BEER-WARE LICENSE" (Revision 42):
<https://github.com/HashakGik> wrote this file.  As long as you retain this notice you can do whatever you want
with this stuff. If we meet some day, and you think this stuff is worth it, you can buy me a beer in return.
"""

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import tqdm.auto as tqdm
import signal # Note: this will not work for non-UNIX systems, as we rely on SIGALRM to detect timeouts.
import time

import torcheval.metrics
from metrics import StdDev, MostProbableMetric, RandomMetric, grad_norm

# Use this file for your training logic. We usually need highly-customized training loops, so a standardized interface
# like Lightning does not suit our needs.

def timeout_handler(signum, frame):
    raise TimeoutError()

def sigint_handler(signum, frame):
    raise KeyboardInterrupt()

def train(net, train_ds, val_ds, test_ds, rng, opts):
    """
    Train the model on the provided datasets. If some events are triggered during training, it collects them on a set of tags.
    These events can be errors (NaN loss, infinite gradients, etc.), warnings (e.g., vanishing gradients, overfitting),
    interrupts (e.g., timeout, user abort, etc.), or informative (e.g., successful run).
    Depending on the experimental setting, some may be more important than others.
    :param net: torch.nn.Module to train.
    :param train_ds: Train dataset.
    :param val_ds: Validation dataset.
    :param test_ds: Test dataset.
    :param opts: Dictionary of hyper-parameters.
    :param rng: Seeded numpy.random.Generator.
    :return: Tuple: (history: list of metrics evaluated for each epoch, return_tags: set of events triggered during training).
    """

    signal.signal(signal.SIGALRM, timeout_handler) # NOTE: this works only on UNIX systems! Windows does not have SIGALRM.
    signal.signal(signal.SIGINT, sigint_handler)

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
    return_tags = set()
    ok = True # True for successful completion and completion with warnings, False for errors.

    # Define here your metrics.
    metrics = {
        s: {
            "accuracy": torcheval.metrics.MulticlassAccuracy(device=opts["device"]),
            "f1": torcheval.metrics.MulticlassF1Score(device=opts["device"]),
        } for s in ["train", "val", "test"]
    }

    baselines = {k: {
            "rnd": {
                "accuracy": RandomMetric(torcheval.metrics.MulticlassAccuracy, opts["device"]),
                "f1": RandomMetric(torcheval.metrics.MulticlassF1Score, opts["device"])
            },
            "mp": {
                "accuracy": MostProbableMetric(torcheval.metrics.MulticlassAccuracy, opts["device"]),
                "f1": MostProbableMetric(torcheval.metrics.MulticlassF1Score, opts["device"])
            }
        } for k in ["train", "val", "test"]
    }


    # Additional metrics.
    metrics["train"]["loss"] = torcheval.metrics.Mean(device=opts["device"])
    metrics["train"]["avg_grad_norm"] = torcheval.metrics.Mean(device=opts["device"])
    metrics["train"]["std_grad_norm"] = StdDev(device=opts["device"])


    for e in tqdm.trange(opts["epochs"], position=0, desc="epoch", disable=opts["verbose"] < 1):
        for v in metrics.values():
            for v2 in v.values():
                v2.reset()

        # To assess generalization, replace computed histograms for validation and test set with training set.
        for k in ["val", "test"]:
            for k2, v in baselines[k]["mp"].items():
                v.set_histogram(baselines["train"]["mp"][k2].get_histogram())

        net.train()

        # Set epoch timeout in minutes. This time includes both training and evaluation stages.
        if opts["epoch_timeout"] > 0:
            signal.alarm(opts["epoch_timeout"] * 60)

        try:
            start_time = time.time()
            # Train the model.
            for batch in (bar := tqdm.tqdm(train_dl, position=1, desc="batch", leave=False, ncols=0, disable=opts["verbose"] < 2)):
                loss, ok, step_tags = train_step(net, optimizer, batch, metrics, baselines, opts)

                return_tags.update(step_tags)
                # If the timer has expired, abort training.
                if not ok:
                    break

                bar.set_postfix_str(
                    "Loss: {:.02f}, avg_Loss: {:.02f}, avg_dLoss: {:.02f}, std_dLoss: {:.02f}, Accuracy: {:.02f}, F1: {:.02f}".format(
                        loss, metrics["train"]["loss"].compute(), metrics["train"]["avg_grad_norm"].compute(),
                        metrics["train"]["std_grad_norm"].compute(), metrics["train"]["accuracy"].compute(),
                        metrics["train"]["f1"].compute()
                    )
                )

            times = {"train": time.time() - start_time}
            with torch.no_grad():
                net.eval()
                # Evaluate the model. If the validation set needs to be used to perform some calibration, split this loop into two.
                for split, dl in {"val": val_dl, "test": test_dl}.items():
                    start_time = time.time()
                    for batch in (bar := tqdm.tqdm(dl, position=1, desc="batch", leave=False, ncols=0, disable=opts["verbose"] < 2)):
                        eval_step(net, batch, metrics, baselines, split, opts)

                        # If the timer has expired, abort evaluation.
                        if not ok:
                            break
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
                epoch_stats["{}/time".format(split)] = times[split]
                for k, v in metrics[split].items():
                    epoch_stats["{}/{}".format(split, k)] = v.compute().item()

            # "Generalization" metrics: train-test and train/test for every recorded metric.
            for k in metrics["test"].keys():
                epoch_stats["delta/{}".format(k)] = epoch_stats["train/{}".format(k)] - epoch_stats["test/{}".format(k)]
                if epoch_stats["delta/{}".format(k)] > opts["overfitting_threshold"]:
                    return_tags.add("Overfitting {} (learning)".format(k))

                if epoch_stats["test/{}".format(k)] != 0:
                    epoch_stats["ratio/{}".format(k)] = epoch_stats["train/{}".format(k)] / epoch_stats[
                        "test/{}".format(k)]
                else:
                    epoch_stats["ratio/{}".format(k)] = 0.0

            # At the end of the first epoch, freeze random baseline metrics.
            for k, v in baselines.items():
                for v2 in v["rnd"].values():
                    v2.freeze()

            history.append(epoch_stats)

        # If the timer has expired, abort training.
        except TimeoutError:
            return_tags.add("Timeout")
            ok = False

        # If the user presses Ctrl+C, abort training. It does not work when W&B is running in sweep mode, because
        # the wrapper will catch the signal before this exception.
        except KeyboardInterrupt:
            return_tags.add("User abort")
            ok = False

        # Whatever happens, disable the timer at the end.
        finally:
            signal.alarm(0)

        if not ok:
            break

    if ok:
        # Add baselines to the history.
        for h in history:
            for split, v in baselines.items():
                for b, v2 in v.items():
                    for m, v3 in v2.items():
                        h["{}/{}_{}".format(split, b, m)] = v3.compute().item()

        for k in metrics["test"].keys():
            if history[-1]["delta/{}".format(k)] > opts["overfitting_threshold"]:
                return_tags.add("Overfitting {} (end)".format(k))
                
            for s in ["train", "val", "test"]:
                if k in baselines[s]["rnd"] and history[-1]["{}/{}".format(s, k)] <= history[-1]["{}/rnd_{}".format(s, k)] + opts["rnd_threshold"]:
                    return_tags.add("Random guessing {} ({})".format(k, s))
                if k in baselines[s]["mp"] and history[-1]["{}/{}".format(s, k)] <= history[-1]["{}/mp_{}".format(s, k)] + opts["mp_threshold"]:
                    return_tags.add("Most probable guessing {} ({})".format(k, s))

        return_tags.add("Success")

    return history, return_tags

def train_step(net, optimizer, batch, metrics, baselines, opts):
    """
    Single training step.
    :param net: torch.nn.Module to train.
    :param optimizer: torch.optim optimizer.
    :param batch: Input batch.
    :param metrics: Metrics to update at the end of the step.
    :param opts: Dictionary of hyper-parameters.
    :return: Tuple (loss: value of the loss, ok: whether training should continue, ret_tags: events triggered by this batch)
    """
    ok = True
    ret_tags = set()
    img, label = batch

    img = img.to(opts["device"])
    label = label.to(opts["device"])

    pred = net(img)

    loss = opts["supervision_lambda"] * F.cross_entropy(pred, label)
    # loss += opts["other_lambda"] * other_loss()

    if torch.isnan(loss):
        ok = False
        ret_tags.add("NaN loss")
    elif torch.isposinf(loss):
        ok = False
        ret_tags.add("Inf loss")
    else:
        optimizer.zero_grad()
        loss.backward()

    metrics["train"]["loss"].update(loss)
    norms = grad_norm(list(net.parameters()))

    if torch.isnan(norms):
        ok = False
        ret_tags.add("NaN gradient")
    else:
        # Avoid NaN weights, by stepping only if gradient is finite.
        optimizer.step()

    if norms < opts["vanishing_threshold"]:
        ret_tags.add("Vanishing gradient")
    elif norms > opts["exploding_threshold"]:
        ret_tags.add("Exploding gradient")

    metrics["train"]["avg_grad_norm"].update(norms)
    metrics["train"]["std_grad_norm"].update(norms)

    if metrics["train"]["std_grad_norm"].compute() > opts["std_threshold"]:
        ret_tags.add("High StdDev")

    # Update metrics.
    metrics["train"]["accuracy"].update(pred, label)
    metrics["train"]["f1"].update(pred, label)

    baselines["train"]["rnd"]["accuracy"].update(pred, label)
    baselines["train"]["rnd"]["f1"].update(pred, label)
    baselines["train"]["mp"]["accuracy"].update(pred, label)
    baselines["train"]["mp"]["f1"].update(pred, label)

    return loss, ok, ret_tags

def eval_step(net, batch, metrics, baselines, split, opts):
    """
    Single evaluation step.
    :param net: torch.nn.Module to evaluate.
    :param batch: Input batch.
    :param metrics: Metrics to update.
    :param baselines: Baseline metrics to update.
    :param split: Dataset split.
    :param opts: Dictionary of hyper-parameters.
    """
    img, label = batch

    img = img.to(opts["device"])
    label = label.to(opts["device"])

    pred = net(img)


    # In this simple example metrics are updated with the same tensors, so we could use a for loop.
    # In general different metrics may be updated by different predictions.
    metrics[split]["accuracy"].update(pred, label)
    metrics[split]["f1"].update(pred, label)

    baselines[split]["rnd"]["accuracy"].update(pred, label)
    baselines[split]["rnd"]["f1"].update(pred, label)
    baselines[split]["mp"]["accuracy"].update(pred, label)
    baselines[split]["mp"]["f1"].update(pred, label)