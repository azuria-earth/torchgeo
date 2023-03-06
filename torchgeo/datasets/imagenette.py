# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Imagenette dataset."""

import os
from typing import Callable, Dict, Optional, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from .geo import NonGeoClassificationDataset
from .utils import check_integrity, download_url, extract_archive, rasterio_loader



class Imagenette(NonGeoClassificationDataset):
    """IMAGENETTE dataset.

    magenette is a subset of 10 easily classified classes from Imagenet 
    (tench, English springer, cassette player, chain saw, church, French horn, garbage truck,
    gas pump, golf ball, parachute).

    Dataset format:

    * rasters are 3-channel JPEG
    * labels are values in the range [0,7]

    Dataset classes:

    * tench
    * English springer
    * cassette player
    * chain saw
    * church
    * French horn
    * garbage truck
    * gas pump
    * golf ball
    * parachute

    This dataset uses the train/val/test splits defined in the "In-domain representation
    learning for remote sensing" paper:

    * https://arxiv.org/abs/1911.06721

    """

    # For some reason the class directories are actually nested in this directory
    base_dir = os.path.join("images")

    splits = ["train", "val", "test"]


    RGB_BANDS = (
        "B01",
        "B02",
        "B03",
    )

    #RGB_BANDS = ("B01", "B02", "B03")

    BAND_SETS = {"all": RGB_BANDS}

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: Sequence[str] = BAND_SETS["all"],
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new EuroSAT dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            bands: a sequence of band names to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match

        .. versionadded:: 0.3
           The *bands* parameter.
        """
        self.root = root
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        assert split in ["train", "val", "test"]

        self.bands = bands
        self._verify()

        valid_fns = set()
        print('La ou ya les .txt', self.root)
        # if split == 'train':
        #     split = 'train_64' #2 samples per class

        with open(os.path.join(self.root, f"imagenette-{split}.txt")) as f:
            for fn in f:
                print('fn[:-1]', fn[:-1])
                valid_fns.add(fn[:-1])
        
        is_in_split: Callable[[str], bool] = lambda x: os.path.basename(x) in valid_fns

        super().__init__(
            root=os.path.join(self.root, self.base_dir),
            transforms=transforms,
            loader=rasterio_loader,
            is_valid_file=is_in_split,
        )
        

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return
        Returns:
            data and label at that index
        """
        image, label = self._load_image(index)
        
        sample = {"index": index, "image": image, "label": label}

        if self.transforms is not None:
          
            sample = self.transforms(sample)

        return sample

    

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the files already exist

        print('self.root', self.root)
        print('self.base_dir', self.base_dir)
        print('os.path.join(self.root, self.base_dir)', os.path.join(self.root, self.base_dir))
        if os.path.exists(os.path.join(self.root, self.base_dir)):
            return

        raise RuntimeError("Dataset not found in `root` directory")

      
    
    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`NonGeoClassificationDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            ValueError: if RGB bands are not found in dataset

        .. versionadded:: 0.2
        """
        rgb_indices = []
        for band in self.RGB_BANDS:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError("Dataset doesn't contain some of the RGB bands")

        image = np.take(sample["image"].numpy(), indices=rgb_indices, axis=0)
        image = np.rollaxis(image, 0, 3)
        image = np.clip(image / 3000, 0, 1)

        label = cast(int, sample["label"].item())
        label_class = self.classes[label]

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = cast(int, sample["prediction"].item())
            prediction_class = self.classes[prediction]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis("off")
        if show_titles:
            title = f"Label: {label_class}"
            if showing_predictions:
                title += f"\nPrediction: {prediction_class}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
