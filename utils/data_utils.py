# utils/data_utils.py

import os
import torch
import rasterio
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# Import constants from config
from config import ISPRS_COLORMAP, IGNORE_INDEX


def read_isprs_images(image_path, label_path):
  img_file_list = os.listdir(image_path)
  lab_file_list = os.listdir(label_path)

  img_file_list = [image_path+'/'+i for i in img_file_list]
  img_file_list.sort()
  lab_file_list = [label_path+'/'+i for i in lab_file_list]
  lab_file_list.sort()

  img, lab = [], []
  for img_file, lab_file in zip(img_file_list, lab_file_list):
    with rasterio.open(img_file) as src:
      img.append(torch.tensor(src.read(), dtype=torch.float32))

    with rasterio.open(lab_file) as src:
      lab.append(torch.tensor(src.read(), dtype=torch.long))

  return img[:1000], lab[:1000]

class ISPRSDataset(torch.utils.data.Dataset):
  """For loading the ISPRS dataset"""
  def __init__(self, images, labels):
    self.img_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    self.mask_transform = transforms.Compose([
          transforms.Resize((300, 300), interpolation=Image.NEAREST)])

    self.images = images
    self.labels = labels

  def __getitem__(self, idx):
    image = self.images[idx]
    label = self.labels[idx]

    if self.img_transform:
      image = self.img_transform(image)
      if image.ndim == 2:
          image = image.unsqueeze(0).repeat(3, 1, 1)

    if self.mask_transform:
      label_pil_for_resize = Image.fromarray(label.cpu().numpy().astype(np.uint8), mode='L')
      label_pil_resized = self.mask_transform(label_pil_for_resize)
      label = torch.from_numpy(np.array(label_pil_resized)).long()
    return image, label.long()

  def __len__(self):
    return len(self.images)
  
def voc_colormap2label():
  """Build a mapping from RGB to VOC category index (labels)"""
  colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
  for i, colormap in enumerate(ISPRS_COLORMAP):
      colormap2label[
          (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
  return colormap2label

# Colormap is the RGB value in the image, which is converted into the corresponding label value
def voc_label_indices(colormap, colormap2label):
  """Map RGB values in VOC labels to their category indices"""
  colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
  idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
          + colormap[:, :, 2])
  return colormap2label[idx]

def worker_init_fn(worker_id):
    """
    Ensures reproducibility for DataLoader workers.
    """
    import random
    import numpy
    # Use a different seed for each worker
    seed = torch.initial_seed() % (2**32 - 1)
    random.seed(seed + worker_id)
    numpy.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + worker_id)