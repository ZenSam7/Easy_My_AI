from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from random import shuffle
from typing import Generator, Tuple
import zlib

import numpy as np
import requests


LabeledImage = Tuple[np.array, np.array]


@dataclass(frozen=True)
class Dataset:
    images: np.array
    labels: np.array

    def __post_init__(self) -> None:
        assert len(self.images) == len(self.labels)

    def __len__(self) -> None:
        return len(self.images)

    def __getitem__(self, index: int) -> LabeledImage:
        return self.images[index], self.labels[index]

    def __setitem__(self, index: int, value: LabeledImage) -> None:
        self.images[index] = value[0]
        self.labels[index] = value[1]

    def minibatches(self, batch_size: int) -> Generator[Dataset, None, None]:
        """Yield chunks of size batch_size from shuffled images and labels."""
        shuffle(self)
        for index in range(0, len(self), batch_size):
            yield self[index:index + batch_size]


@dataclass(frozen=True)
class MNIST:
    url_base: str = 'http://yann.lecun.com/exdb/mnist'
    cache_dir: Path = Path('/', 'tmp', 'mnist')

    @cached_property
    def train_set(self) -> Dataset:
        return Dataset(
            self.load_images('train-images-idx3-ubyte.gz'),
            self.load_labels('train-labels-idx1-ubyte.gz'),
        )

    @cached_property
    def test_set(self) -> Dataset:
        return Dataset(
            self.load_images('t10k-images-idx3-ubyte.gz'),
            self.load_labels('t10k-labels-idx1-ubyte.gz'),
        )

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def raw_bytes(self, filename: str, offset: int) -> np.array:
        cached_file = Path(self.cache_dir, filename)
        if not cached_file.exists():
            with requests.get(f'{self.url_base}/{filename}') as resp:
                resp.raise_for_status()
                with open(cached_file, 'wb') as f:
                    f.write(resp.content)
        with open(cached_file, 'rb') as f:
            raw_bytes = zlib.decompress(f.read(), 15 + 32)
        return np.frombuffer(raw_bytes, '>B', offset=offset)

    def load_images(self, filename: str) -> np.array:
        """Return zero-to-one scaled images as rows of a matrix."""
        return self.raw_bytes(filename, offset=16).reshape(-1, 784) / 255

    def load_labels(self, filename: str) -> np.array:
        """Return one-hot encoded class labels as rows of a matrix."""
        return np.eye(10)[self.raw_bytes(filename, offset=8)]
