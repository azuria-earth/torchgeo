# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""OPSSAT datamodule."""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize 
from ..datasets import OPSSAT
import albumentations as A


class OPSSATDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the OPSSAT dataset.

    Uses the train/val/test splits from the dataset.

    .. versionadded:: 0.2
    """

    band_means = torch.tensor(
        [
            60.04637211,
            101.57636894,
            104.77283831,
        ]
    )

    band_stds = torch.tensor(
        [
            37.10114102,
            54.46382011,
            52.2372836, 
        ]
    )

    def __init__(
        self, root_dir: str, root_dir_images_nonhisteq: str, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a LightningDataModule for EuroSAT based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the EuroSAT Dataset classes
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.root_dir_images_nonhisteq = root_dir_images_nonhisteq
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.norm = Normalize(self.band_means, self.band_stds)
        self.T  = A.Compose(
        [
            A.RandomResizedCrop(224,224),
            A.ShiftScaleRotate(p=0.2),
        ]
        )
       
       

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: input image dictionary

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        sample["image"] = self.norm(sample["image"])
        return sample

    def augmentation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: input image dictionary

        Returns:
            preprocessed sample
        """
        import numpy as np

    
        Transform = self.T
        x = sample["image"]
        img = np.array(x.permute(1,2,0).cpu())
        aug = Transform(image=img)["image"].transpose(2,0,1)
        New_X = torch.from_numpy(aug)

        sample["image"] = New_X

        return sample


    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        OPSSAT(self.root_dir,self.root_dir_images_nonhisteq)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        transforms_train = Compose([self.preprocess, self.augmentation])
        transforms = Compose([self.preprocess])

        self.train_dataset = OPSSAT(self.root_dir,self.root_dir_images_nonhisteq, "train", transforms=transforms)
        self.val_dataset = OPSSAT(self.root_dir,self.root_dir_images_nonhisteq, "val", transforms=transforms)
        self.test_dataset = OPSSAT(self.root_dir,self.root_dir_images_nonhisteq, "test", transforms=transforms)
        

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing.

        Returns:
            testing data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        """Run :meth:`torchgeo.datasets.OPSSAT.plot`."""
        return self.val_dataset.plot(*args, **kwargs)
