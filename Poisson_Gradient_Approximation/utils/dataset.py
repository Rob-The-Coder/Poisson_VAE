import torch
import torchvision
import pandas as pd

from PIL import Image
from pathlib import Path

class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, img_dir, img_names, transform=None):
    self.img_dir = img_dir
    self.transform = transform
    self.img_names = img_names

  def __len__(self):
    return len(self.img_names)

  def __getitem__(self, idx):
    img_path = Path(self.img_dir) / self.img_names[idx]
    image = Image.open(img_path).convert('RGB') # VAE expects 3 channels
    if self.transform:
      image = self.transform(image)
    return image, 0 # Return a dummy label for compatibility with DataLoader

  @staticmethod
  def get_dataloaders(height, width, batch_size, path):
    transform = torchvision.transforms.Compose([
      torchvision.transforms.CenterCrop(178),
      torchvision.transforms.Resize((height, width)),
      torchvision.transforms.RandomHorizontalFlip(),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(0.5, 0.5)
    ])

    img_folder_path = Path(path) / "img_align_celeba" / "img_align_celeba"
    partition_df = pd.read_csv(Path(path) / "list_eval_partition.csv")

    train_partition = partition_df[partition_df['partition'] == 0]['image_id'].tolist()
    valid_partition = partition_df[partition_df['partition'] == 1]['image_id'].tolist()

    train_set = CustomDataset(img_folder_path, train_partition, transform=transform)
    valid_set = CustomDataset(img_folder_path, valid_partition, transform=transform)
    # test_partition = partition_df[partition_df['partition'] == 2]['image_id'].tolist() # Not used to this moment

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)

    return train_loader, valid_loader
