"""
"THE BEER-WARE LICENSE" (Revision 42):
<https://github.com/HashakGik> wrote this file.  As long as you retain this notice you can do whatever you want
with this stuff. If we meet some day, and you think this stuff is worth it, you can buy me a beer in return.
"""

import torch
import torcheval

# Put in this file any custom metrics.

class Variance(torcheval.metrics.Metric[torch.Tensor]):
    """
    Variance metric. Implementation of: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    def __init__(self, device=None):
        super().__init__(device=device)
        self._add_state("n", torch.tensor(0.0, device=self.device, dtype=torch.float64))
        self._add_state("avg", torch.tensor(0.0, device=self.device, dtype=torch.float64))
        self._add_state("mse", torch.tensor(0.0, device=self.device, dtype=torch.float64))
        pass

    def reset(self):
        self.n = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        self.avg = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        self.mse = torch.tensor(0.0, device=self.device, dtype=torch.float64)

    @torch.inference_mode()
    def compute(self):
        if self.n > 1:
            return self.mse / (self.n - 1)
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
    Call this AFTER backward()
    :param parameters: List of model parameters.
    :return: The gradient norm.
    """
    if len(parameters) == 0:
        total_norm = 0.0
    else:
        device = parameters[0].grad.device
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters if p.grad is not None]),
            2.0)
    return total_norm