"""
"THE BEER-WARE LICENSE" (Revision 42):
<https://github.com/HashakGik> wrote this file.  As long as you retain this notice you can do whatever you want
with this stuff. If we meet some day, and you think this stuff is worth it, you can buy me a beer in return.
"""

import torch
import torcheval
import torch.nn.functional as F

class RandomMetric(torcheval.metrics.Metric[torch.Tensor]):
    """
    Random-guessing wrapper. It takes an arbitrary metric, and updates it with a random prediction.
    """
    def __init__(self, metric, device=None):
        """
        :param metric: torcheval.metrics.Metric updated with a random prediction.
        :param device: torch.device.
        """
        self.metric = metric(device=device)
        self._frozen = False

    def freeze(self):
        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    def reset(self):
        if not self._frozen:
            self.metric.reset()

    @torch.inference_mode()
    def compute(self):
        return self.metric.compute()

    @torch.inference_mode()
    def update(self, input, target):
        if not self._frozen:
            if input.dim() == 1: # The metric measures probabilities.
                rnd_input = torch.rand(size=(input.size(0),), device=input.device)
            else: # The metric measures classes.
                rnd_input = F.one_hot(torch.randint(input.size(-1), size=(input.size(0),), device=input.device), num_classes=input.size(-1))

            self.metric.update(rnd_input, target)
        return self

    @torch.inference_mode()
    def merge_state(self, metrics):
        for metric in metrics:
            if metric.inputs:
                self.inputs.append(
                    torch.cat(metric.inputs, dim=metric.dim).to(self.device)
                )
        return self


class MostProbableMetric(torcheval.metrics.Metric[torch.Tensor]):
    """
    Most-probable-guessing wrapper. It takes an arbitrary *classification metric* (i.e., inputs are scalars or categoricals),
    and updates it by using the most probable class observed so far.
    To be fair this should always use the train distribution, not the observations of the current split.
    One way to guarantee this, is to ignore results on the first epoch and at the beginning of the second,
    copy the training metric histogram to other splits and freeze it.

    For streaming settings, it may be more reasonable to let the histograms be updated indefinitely.
    """

    def __init__(self, metric, device=None):
        """
        :param metric: torcheval.metrics.Metric updated with the most likely prediction up to that point.
        :param device: torch.device.
        """
        self.metric = metric(device=device)
        self._frozen = False
        self._histogram = None
        self._metric_value = torch.zeros(1)

    def reset_histogram(self):
        self._histogram = None

    def set_histogram(self, h, reset_metric=True, freeze_on_set=True):
        """
        Copies the internal histogram from another source.
        :param h: The histogram to copy.
        :param reset_metric: If True, resets the metric statistics.
        :param freeze_on_set: If True and the histogram is not empty, prevents it from being updated further.
        """
        self._histogram = h
        if reset_metric:
            self.metric.reset()
        if freeze_on_set and h is not None:
            self._frozen = True

    def freeze(self):
        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    def get_histogram(self):
        return self._histogram

    def reset(self):
        self.metric.reset()


    @torch.inference_mode()
    def compute(self):
        return self.metric.compute()


    @torch.inference_mode()
    def update(self, input, target):
        # Update histograms.
        if not self._frozen:
            if input.dim() == 1:
                if self._histogram is None:
                    self._histogram = torch.zeros(2, device=input.device)

                self._histogram[0] += torch.count_nonzero(target <= 0.5)
                self._histogram[1] += torch.count_nonzero(target > 0.5)
            else:
                if self._histogram is None:
                    self._histogram = torch.zeros((input.size(-1),), device=input.device)

                batch_counts = torch.bincount(target.to(torch.int64), minlength=self._histogram.size(0))
                self._histogram += batch_counts

        # Update metric.
        if input.dim() == 1:  # The metric measures probabilities.
            mp_input = torch.argmax(self._histogram).unsqueeze(0).expand(target.size(0))
        else:  # The metric measures categories.
            mp_input = F.one_hot(torch.argmax(self._histogram), num_classes=input.size(-1)).unsqueeze(0).expand(target.size(0), -1)

        self.metric.update(mp_input, target)


        return self

    @torch.inference_mode()
    def merge_state(self, metrics):
        for metric in metrics:
            if metric.inputs:
                self.inputs.append(
                    torch.cat(metric.inputs, dim=metric.dim).to(self.device)
                )
        return self

class StdDev(torcheval.metrics.Metric[torch.Tensor]):
    """
    Standard deviation metric. Implementation of: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    def __init__(self, device=None):
        super().__init__(device=device)
        self._add_state("n", torch.tensor(0.0, device=self.device, dtype=torch.float64))
        self._add_state("avg", torch.tensor(0.0, device=self.device, dtype=torch.float64))
        self._add_state("mse", torch.tensor(0.0, device=self.device, dtype=torch.float64))

    def reset(self):
        self.n = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        self.avg = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        self.mse = torch.tensor(0.0, device=self.device, dtype=torch.float64)

    @torch.inference_mode()
    def compute(self):
        if self.n > 1:
            return torch.sqrt(self.mse / (self.n - 1))
        else:
            return self.n

    @torch.inference_mode()
    def update(self, input):
        if input.dim() == 0:
            input = input.unsqueeze(0)
        old_n = self.n
        new_n = input.size(0)

        if new_n >= 1:
            old_avg = self.avg
            new_avg = torch.mean(input)

            self.n += new_n
            self.avg = (old_n * old_avg + new_n * new_avg) / self.n

            delta = new_avg - old_avg

            new_mse = torch.sum((input - new_avg) ** 2)
            self.mse += new_mse + delta ** 2 * old_n * new_n / self.n

        return self

    @torch.inference_mode()
    def merge_state(self, metrics):
        for metric in metrics:
            if metric.inputs:
                self.inputs.append(
                    torch.cat(metric.inputs, dim=metric.dim).to(self.device)
                )
        return self


def grad_norm(parameters):
    """
    Compute the norm of gradients, following: https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961
    Call this AFTER backward(). If you are trying to avoid updating with NaN gradients,
    call optimizer.step() AFTER checking this.
    :param parameters: List of model parameters.
    :return: The gradient norm.
    """
    total_norm = torch.zeros(1)

    if len(parameters) > 0:
        device = parameters[0].device
        tmp = [torch.norm(p.grad.detach(), 2).to(device) for p in parameters if p.grad is not None]
        if len(tmp) > 0:
            total_norm = torch.norm(torch.stack(tmp), 2.0)

    return total_norm