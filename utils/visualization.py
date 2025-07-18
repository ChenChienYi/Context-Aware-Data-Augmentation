# utils/visualization.py

import matplotlib.pyplot as plt
import numpy as np
import torch

# Import constants from config and functions from data_utils
from config import ISPRS_COLORMAP
from utils.data_utils import voc_colormap2label, voc_label_indices

def normalize_to_0_1(img_np):
    min_val = np.min(arr_np)
    max_val = np.max(arr_np)
    if max_val > min_val:
        arr_np_scaled = (arr_np - min_val) / (max_val - min_val)
    else:
        # If all values are the same, result in all zeros
        arr_np_scaled = np.zeros_like(arr_np)
    return arr_np_scaled

def mask_to_rgb(mask, colormap):
    """Converts a class index mask (H, W) to an RGB image (H, W, 3) using a colormap."""
    if mask.ndim != 2:
        raise ValueError("Input mask to mask_to_rgb must be 2D (H, W)")

    h, w = mask.shape
    output_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    if 'ISPRS_COLORMAP' in globals():
        colormap_to_use = ISPRS_COLORMAP
    else:
        print("Warning: ISPRS_COLORMAP not found. Using a default grayscale colormap.")
        colormap_to_use = [[i, i, i] for i in range(len(np.unique(mask)))]

    for class_id, color in enumerate(colormap_to_use):
        output_rgb[mask == class_id] = color
    return output_rgb

def display_image_and_label(images_list, labels_list, index_to_display = None, is_class_id_label=False):
  """
  Displays an image and its corresponding label mask from lists of PyTorch tensors.
  """

  if not (0 <= index_to_display < len(images_list)):
    print(f"Error: Index {index_to_display} is out of bounds for images_list (size {len(images_list)}).")
    return
  if not (0 <= index_to_display < len(labels_list)):
    print(f"Error: Index {index_to_display} is out of bounds for labels_list (size {len(labels_list)}).")
    return

  img_tensor = images_list[index_to_display]
  label_tensor = labels_list[index_to_display]

  img_np = img_tensor.numpy()
  img_np = np.transpose(img_np, (1, 2, 0))

  if not is_class_id_label:
    label_indices_np = voc_label_indices(label_tensor.cpu(), voc_colormap2label()).numpy()
    label_np_processed = label_indices_np
  else:
    label_np_processed = label_tensor.numpy()

  def normalize_to_0_1(img_np):
      min_val = np.min(img_np)
      max_val = np.max(img_np)
      if max_val > min_val:
          img_np_scaled = (img_np - min_val) / (max_val - min_val)
      else:
          img_np_scaled = np.zeros_like(img_np)

      return img_np_scaled

  img_np_normalized = normalize_to_0_1(img_np)
  def mask_to_rgb(mask, colormap):
      """Converts a class index mask (H, W) to an RGB image (H, W, 3) using a colormap."""
      if mask.ndim != 2:
          raise ValueError("Input mask to mask_to_rgb must be 2D (H, W)")

      h, w = mask.shape
      output_rgb = np.zeros((h, w, 3), dtype=np.uint8)
      if 'ISPRS_COLORMAP' in globals():
            colormap_to_use = ISPRS_COLORMAP
      else:
            print("Warning: ISPRS_COLORMAP not found. Using a default grayscale colormap.")
            colormap_to_use = [[i, i, i] for i in range(len(np.unique(mask)))]

      for class_id, color in enumerate(colormap_to_use):
          output_rgb[mask == class_id] = color
      return output_rgb

  label_display_rgb = mask_to_rgb(label_np_processed, ISPRS_COLORMAP)

  plt.figure(figsize=(12, 6))
  plt.subplot(1, 2, 1)
  plt.imshow(img_np_normalized)
  plt.title(f"Image (Index: {index_to_display})")
  plt.axis('off')

  plt.subplot(1, 2, 2)
  plt.imshow(label_display_rgb)
  plt.title(f"Label (Index: {index_to_display})")
  plt.axis('off')

  plt.tight_layout()
  plt.show()