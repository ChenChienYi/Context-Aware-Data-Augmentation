import os
import torch
import random
import numpy as np
import itertools
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import kagglehub # If running on Kaggle

# Import functions and classes from your custom modules
from config import (
    ISPRS_COLORMAP, ISPRS_CLASSES, NUM_CLASSES, IGNORE_INDEX,
    HPARAM_COMBINATIONS, N_EPOCHS, PATIENCE, SEED,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)
from utils.data_utils import read_isprs_images, voc_colormap2label, voc_label_indices, ISPRSDataset, worker_init_fn
from utils.augmentation import ContextAwareAugmenter
from models.fcn_resnet import initialize_fcn_resnet50_model
from training.trainer import train_val, calculate_iou_mask, train_one_epoch, validate_one_epoch
from utils.visualization import display_image_and_label, normalize_to_0_1, mask_to_rgb

def set_reproducibility(seed):
    """Sets seeds for reproducibility across different libraries."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set random seed to {seed} for reproducibility.")

def main():
    # Set reproducibility
    set_reproducibility(SEED)

    # 0. Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 1. Loading data
    # Adjust path if not running on Kaggle
    try:
        path = kagglehub.dataset_download("jahidhasan66/isprs-potsdam")
        data_dir = os.path.join(path, 'patches')
    except Exception as e:
        print(f"KaggleHub download failed or not on Kaggle: {e}")
        # Fallback for local development: adjust this path to your dataset location
        data_dir = './data/isprs_potsdam_patches' # Example local path
        if not os.path.exists(data_dir):
            print(f"Error: Data directory '{data_dir}' not found. Please download the dataset.")
            return

    image_path = os.path.join(data_dir, 'Images')
    label_path = os.path.join(data_dir, 'Labels')

    print(f"Loading images from: {image_path}")
    print(f"Loading labels from: {label_path}")
    
    # Load raw images and RGB labels
    images, labels_rgb = read_isprs_images(image_path, label_path, limit=1000)
    print(f"Loaded {len(images)} raw images and {len(labels_rgb)} RGB labels.")

    # Convert RGB labels to class ID labels (H, W)
    colormap2label_map = voc_colormap2label()
    labels_class_ids = [voc_label_indices(label, colormap2label_map) for label in labels_rgb]
    print(f"Converted {len(labels_class_ids)} RGB labels to class ID labels (H, W).")

    # 2. Split data
    total_samples = min(len(images), len(labels_class_ids))
    train_count = int(total_samples * TRAIN_RATIO)
    val_count = int(total_samples * VAL_RATIO)
    test_count = total_samples - train_count - val_count

    train_images_orig = images[:train_count]
    train_labels_orig = labels_class_ids[:train_count]

    val_images = images[train_count : train_count + val_count]
    val_labels = labels_class_ids[train_count : train_count + val_count]

    test_images = images[train_count + val_count : total_samples]
    test_labels = labels_class_ids[train_count + val_count : total_samples]

    print(f"Original train_images count: {len(train_images_orig)}, train_labels count: {len(train_labels_orig)}")
    print(f"Validation images count: {len(val_images)}, val_labels count: {len(val_labels)}")
    print(f"Test images count: {len(test_images)}, test_labels count: {len(test_labels)}")

    # 3. Data augmentation (Context-Aware Augmenter)
    # The augmenter needs original RGB labels for car cropping
    augmenter = ContextAwareAugmenter(images, labels_rgb, ISPRS_COLORMAP, ISPRS_CLASSES)

    num_desired_augmentations = 10
    augmented_images_list = []
    augmented_labels_list = []
    attempts_made = 0

    print(f"\nAttempting to create {num_desired_augmentations} successful augmented images...")
    while len(augmented_images_list) < num_desired_augmentations and attempts_made < 1000:
        attempts_made += 1
        image_index_to_augment = random.randint(0, len(images) - 1) 
        augmented_image, augmented_label = augmenter.augment_image(image_index_to_augment)

        if augmented_image is not None and augmented_label is not None:
            augmented_images_list.append(augmented_image)
            augmented_labels_list.append(augmented_label)
            print(f"Created augmented image {len(augmented_images_list)}/{num_desired_augmentations} (Attempt {attempts_made})")
        # else:
            # print(f"Failed to create augmented image (Attempt {attempts_made}). Trying again...")

    if len(augmented_images_list) == num_desired_augmentations:
        print(f"\nSuccessfully created {num_desired_augmentations} augmented images.")
    else:
        print(f"\nFinished after {attempts_made} attempts. Created {len(augmented_images_list)} out of {num_desired_augmentations} desired augmented images.")
        print("Could not reach the desired number of augmentations. Consider increasing max_attempts or checking augmentation logic.")

    # Extend original training data with augmented data
    train_images = train_images_orig + augmented_images_list
    train_labels = train_labels_orig + augmented_labels_list

    print(f"\nFinal train_images count: {len(train_images)}, train_labels count: {len(train_labels)}")
    print(f"Final val_images count: {len(val_images)}, val_labels count: {len(val_labels)}")
    print(f"Final test_images count: {len(test_images)}, test_labels count: {len(test_labels)}")

    # Create PyTorch Datasets
    train_dataset = ISPRSDataset(train_images, train_labels)
    val_dataset = ISPRSDataset(val_images, val_labels)
    test_dataset = ISPRSDataset(test_images, test_labels)

    # 4. Hyperparameter Search and Training
    base_log_dir = 'runs'
    os.makedirs(base_log_dir, exist_ok=True)

    print("\nStarting hyperparameter search...")
    for i, (lr, bs, opt_name) in enumerate(HPARAM_COMBINATIONS):
        hparams = {
            'learning_rate': lr,
            'batch_size': bs,
            'optimizer': opt_name
        }

        run_name = f"run_{i+1}_lr{lr}_bs{bs}_opt{opt_name}"
        log_path = os.path.join(base_log_dir, run_name)
        writer = SummaryWriter(log_dir=log_path)

        print(f"\n--- Running Experiment {i+1}/{len(HPARAM_COMBINATIONS)}: {hparams} ---")
        
        # Initialize a NEW model for each experiment
        model_fcn = initialize_fcn_resnet50_model(NUM_CLASSES, device)

        # Call the main training-validation function
        _, final_loss, final_iou = train_val(model_fcn, hparams, writer, train_dataset, val_dataset)

        writer.add_hparams(
            hparam_dict=hparams,
            metric_dict={
                'hparam/final_loss': final_loss,
                'hparam/final_iou': final_iou
            }
        )

        writer.close()
        print(f"Experiment {i+1} completed. Final Validation Loss: {final_loss:.4f}, Final Validation IoU: {final_iou:.4f}")

    print("\nHyperparameter search complete. You can view results in TensorBoard.")
    print("To run TensorBoard: `tensorboard --logdir runs` in your terminal.")

if __name__ == "__main__":
    main()