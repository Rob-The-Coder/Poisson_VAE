import torch
import torchvision
import pandas as pd

from PIL import Image
from pathlib import Path

class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, img_dir: Path, img_partition, transform=None):
    self.img_dir = img_dir
    self.transform = transform
    self.img_partition = img_partition

  def __len__(self):
    return len(self.img_partition)

  def __getitem__(self, idx):
    img_path = self.img_dir / self.img_partition[idx]
    image = Image.open(img_path).convert('RGB') # VAE expects 3 channels
    if self.transform:
      image = self.transform(image)
    return image, 0 # Return a dummy label for compatibility with DataLoader

  @staticmethod
  def __get_transform(height, width):
    transform = torchvision.transforms.Compose([
      torchvision.transforms.CenterCrop(178),
      torchvision.transforms.Resize((height, width)),
      torchvision.transforms.RandomHorizontalFlip(),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(0.5, 0.5)
    ])

    return transform

  @staticmethod
  def get_dataloaders(height, width, batch_size, path: Path):
    transform = CustomDataset.__get_transform(height, width)

    img_folder_path = path / "img_align_celeba" / "img_align_celeba"
    partition_df = pd.read_csv(path / "list_eval_partition.csv")

    train_partition = partition_df[partition_df['partition'] == 0]['image_id'].tolist()
    valid_partition = partition_df[partition_df['partition'] == 1]['image_id'].tolist()

    train_set = CustomDataset(img_folder_path, train_partition, transform=transform)
    valid_set = CustomDataset(img_folder_path, valid_partition, transform=transform)
    # test_partition = partition_df[partition_df['partition'] == 2]['image_id'].tolist() # Not used to this moment

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)

    return train_loader, valid_loader

  @staticmethod
  def get_train_set(height, width, path: Path):
    transform = CustomDataset.__get_transform(height, width)

    img_folder_path = path / "img_align_celeba" / "img_align_celeba"
    partition_df = pd.read_csv(path / "list_eval_partition.csv")

    train_partition = partition_df[partition_df['partition']==0]['image_id'].tolist()

    train_set = CustomDataset(img_folder_path, train_partition, transform=transform)

    return train_set