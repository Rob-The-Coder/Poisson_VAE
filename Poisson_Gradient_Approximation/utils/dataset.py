import random

import torch
import pandas as pd
import torchvision.transforms.v2 as T

from torchvision.io import read_image, write_jpeg
from abc import ABC, abstractmethod
from pathlib import Path

class CustomDataset(ABC, torch.utils.data.Dataset):
  @abstractmethod
  def __init__(self, img_dir: Path, img_partition, transform=None):
    pass

  @abstractmethod
  def __len__(self):
    pass

  @abstractmethod
  def __getitem__(self, idx):
    pass

  @staticmethod
  @abstractmethod
  def get_transform(height, width):
    pass

  @staticmethod
  @abstractmethod
  def get_dataloaders(height, width, batch_size, images_dir: Path):
    pass

class CelebA(CustomDataset):
  def __init__(self, img_dir: Path, img_partition, transform=None):
    self.img_dir = img_dir
    self.transform = transform
    self.img_partition = img_partition

  def __len__(self):
    return len(self.img_partition)

  def __getitem__(self, idx):
    img_path = self.img_dir / self.img_partition[idx]
    image = read_image(str(img_path))
    if self.transform:
      image = self.transform(image)
    return image, 0  # Return a dummy label for compatibility with DataLoader

  @staticmethod
  def preprocess_to_disk(images_dir: Path, height: int, width: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = T.Compose([
      T.CenterCrop(178),
      T.Resize((height, width), antialias=True),
    ])

    img_folder = images_dir / "img_align_celeba" / "img_align_celeba"
    for img_name in img_folder.iterdir():
      img = read_image(str(img_name))
      img = transform(img)
      write_jpeg(img, str(output_dir / img_name.name), quality=95)

  @staticmethod
  def get_transform(height, width):
    return T.Compose([
      T.RandomHorizontalFlip(),
      T.ToDtype(torch.float32, scale=True),
      T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

  @staticmethod
  def get_dataloaders(height, width, batch_size, images_dir: Path):
    transform = CelebA.get_transform(height, width)

    img_folder_path = images_dir / "img_align_celeba_resized" / "img_align_celeba_resized"
    if not img_folder_path.exists() or not img_folder_path.is_dir():
      CelebA.preprocess_to_disk(images_dir, height, width, img_folder_path)

    partition_df = pd.read_csv(images_dir / "list_eval_partition.csv")

    train_partition = partition_df[partition_df['partition']==0]['image_id'].tolist()
    valid_partition = partition_df[partition_df['partition']==1]['image_id'].tolist()

    train_set = CelebA(img_folder_path, train_partition, transform=transform)
    valid_set = CelebA(img_folder_path, valid_partition, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True,
                                               num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=True,
                                               num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    return train_loader, valid_loader

  @staticmethod
  def get_train_set(height, width, path: Path):
    transform = CelebA.get_transform(height, width)

    img_folder_path = path / "img_align_celeba_resized" / "img_align_celeba_resized"
    if not img_folder_path.exists() or not img_folder_path.is_dir():
      CelebA.preprocess_to_disk(path, height, width, img_folder_path)

    partition_df = pd.read_csv(path / "list_eval_partition.csv")

    train_partition = partition_df[partition_df['partition']==0]['image_id'].tolist()

    train_set = CelebA(img_folder_path, train_partition, transform=transform)

    return train_set

  @staticmethod
  def get_valid_set(height, width, path: Path):
    transform = CelebA.get_transform(height, width)

    img_folder_path = path / "img_align_celeba_resized" / "img_align_celeba_resized"
    if not img_folder_path.exists() or not img_folder_path.is_dir():
      CelebA.preprocess_to_disk(path, height, width, img_folder_path)

    partition_df = pd.read_csv(path / "list_eval_partition.csv")

    valid_partition = partition_df[partition_df['partition']==1]['image_id'].tolist()

    valid_set = CelebA(img_folder_path, valid_partition, transform=transform)

    return valid_set

  @staticmethod
  def get_attributes(path: Path):
    return pd.read_csv(path / 'list_attr_celeba.csv').replace(-1, 0)

  def get_partition_idx(self, attr_df, attribute: str):
    id_to_idx = {name: i for i, name in enumerate(self.img_partition)}

    pos_ids = attr_df[attr_df[attribute]==1]['image_id'].tolist()
    neg_ids = attr_df[attr_df[attribute]!=1]['image_id'].tolist()

    pos_idx = [id_to_idx[img_id] for img_id in pos_ids if img_id in id_to_idx]
    neg_idx = [id_to_idx[img_id] for img_id in neg_ids if img_id in id_to_idx]

    return pos_idx, neg_idx