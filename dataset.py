"""
"THE BEER-WARE LICENSE" (Revision 42):
<https://github.com/HashakGik> wrote this file.  As long as you retain this notice you can do whatever you want
with this stuff. If we meet some day, and you think this stuff is worth it, you can buy me a beer in return.
"""

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor

#import pandas as pd
#from torchvision.io import read_image

# Use this file for your datasets. We usually deal with non i.i.d. data, so our dataset logic can become very complex.


class MyDataset(Dataset):
    """
    Simple MNIST dataset. This wrapper is not required for such a simple example.
    """
    def __init__(self, opts, split, transform=None):
        """
        :param opts: Dictionary of hyper-parameters.
                     It is bad programming practice, but if we have multiple HPs related to dataset behavior,
                     this is cleaner than the Accumulate & Fire anti-pattern.
                     We often do not need this and replace it with a simple csv path.
        :param split: Dataset split.
        :param transform: Some torchvision.transforms object to apply before returning the sample.
        """
        super().__init__()

        if transform is not None:
            self.transform = Compose([transform, ToTensor()])
        else:
            self.transform = ToTensor()
        self.mnist = datasets.MNIST("tmp", train=(split != "test"), transform=self.transform, download=True)
        self.split = split

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]

        return img, label
