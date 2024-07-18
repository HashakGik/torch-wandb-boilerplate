"""
"THE BEER-WARE LICENSE" (Revision 42):
<https://github.com/HashakGik> wrote this file.  As long as you retain this notice you can do whatever you want
with this stuff. If we meet some day, and you think this stuff is worth it, you can buy me a beer in return.
"""

import torch
import torch.nn.functional as F

# Put in this file your architecture. We usually deal with complex implementations which may be split into multiple files.

class MNISTNet(torch.nn.Module):
    """
    Simple MNIST digit classifier.
    """
    def __init__(self, opts):
        """
        :param opts: Dictionary of hyper-parameters.
                     It is bad programming practice, but if we have multiple HPs related to the architecture,
                     this is cleaner than the Accumulate & Fire anti-pattern.
        """
        super().__init__()

        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = torch.nn.Linear(1024, 1024)
        self.fc2 = torch.nn.Linear(1024, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)

        return x

