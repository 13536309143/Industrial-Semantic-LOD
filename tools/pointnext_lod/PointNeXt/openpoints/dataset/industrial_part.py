import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .build import DATASETS


@DATASETS.register_module()
class IndustrialPartPointCloud(Dataset):
    """Class-folder industrial part point-cloud dataset.

    The converter writes list files with rows:
        relative/path/to/file.npy<TAB>class_index<TAB>class_name
    """

    def __init__(
        self,
        data_dir="../pointcloud_data",
        split="train",
        num_points=4096,
        list_file=None,
        transform=None,
        shuffle_points=True,
    ):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_absolute():
            self.data_dir = (Path.cwd() / self.data_dir).resolve()

        self.split = "test" if split in ["val", "validation", "eval", "evaluation"] else split
        self.num_points = int(num_points)
        self.transform = transform
        self.shuffle_points = bool(shuffle_points)

        list_path = Path(list_file) if list_file is not None else self.data_dir / f"{self.split}_list.txt"
        if not list_path.is_absolute():
            list_path = (self.data_dir / list_path).resolve()
        if not list_path.exists():
            raise FileNotFoundError(f"Missing dataset list file: {list_path}")

        self.classes = self._load_classes()
        self.samples = self._load_samples(list_path)
        if not self.samples:
            raise ValueError(f"No samples found in {list_path}")

        logging.info(
            "Loaded IndustrialPartPointCloud split=%s samples=%d classes=%d data_dir=%s",
            self.split,
            len(self.samples),
            self.num_classes,
            self.data_dir,
        )

    def _load_classes(self):
        classes_path = self.data_dir / "classes.txt"
        if not classes_path.exists():
            return None

        classes = []
        with classes_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.rstrip("\n")
                if not line:
                    continue
                idx_text, class_name = line.split("\t", 1)
                idx = int(idx_text)
                while len(classes) <= idx:
                    classes.append("")
                classes[idx] = class_name
        return classes

    def _load_samples(self, list_path):
        samples = []
        with list_path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, 1):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    raise ValueError(f"Bad line {line_no} in {list_path}: {line!r}")

                rel_path = Path(parts[0])
                label = int(parts[1])
                point_path = self.data_dir / rel_path
                if not point_path.exists():
                    raise FileNotFoundError(f"Point cloud listed but missing: {point_path}")
                samples.append((point_path, label))
        return samples

    def _fit_num_points(self, points):
        if len(points) == self.num_points:
            return points

        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
        else:
            indices = np.random.choice(len(points), self.num_points, replace=True)
        return points[indices]

    def __getitem__(self, index):
        path, label = self.samples[index]
        pointcloud = np.load(path).astype(np.float32, copy=False)
        if pointcloud.ndim != 2 or pointcloud.shape[1] < 3:
            raise ValueError(f"Expected point cloud shaped [N, >=3], got {pointcloud.shape}: {path}")

        pointcloud = pointcloud[:, :3]
        pointcloud = self._fit_num_points(pointcloud)
        if self.split == "train" and self.shuffle_points:
            np.random.shuffle(pointcloud)

        data = {
            "pos": pointcloud,
            "y": np.array(label, dtype=np.int64),
        }
        if self.transform is not None:
            data = self.transform(data)

        data["x"] = data["pos"]
        return data

    def __len__(self):
        return len(self.samples)

    @property
    def num_classes(self):
        if self.classes is not None:
            return len(self.classes)
        return max(label for _, label in self.samples) + 1
