import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Import constants from config
from config import N_EPOCHS, PATIENCE, NUM_CLASSES, IGNORE_INDEX

def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
  """
  Run a full training cycle (epoch)
  """
  model.train()
  running_train_loss = 0.0

  for i, (features, labels) in enumerate(train_loader):
    features, labels = features.to(device), labels.to(device)
    optimizer.zero_grad()
    classes_preds = model(features)['out']
    training_loss = loss_fn(classes_preds, labels)
    training_loss.backward()
    optimizer.step()
    running_train_loss += training_loss.item()

  avg_train_loss = running_train_loss / len(train_loader)
  return model, avg_train_loss

def test_model(model, val_loader, loss_fn, device, num_classes, ignore_index):
  """
  Perform a full validation cycle (epoch)
  """
  model.eval()
  running_val_loss = 0.0
  running_val_iou = 0.0
  num_batches_with_valid_iou = 0

  with torch.no_grad():
    for j, (features, labels) in enumerate(val_loader):
      features, labels = features.to(device), labels.to(device)

      classes_preds = model(features)['out']
      val_loss = loss_fn(classes_preds, labels)
      predicted_labels = torch.argmax(classes_preds, dim=1)
      batch_iou, _ = calculate_iou_mask(
          predicted_labels.cpu().numpy(),
          labels.cpu().numpy(),
          num_classes=num_classes,
          ignore_index=ignore_index
      )

      running_val_loss += val_loss.item()
      if not np.isnan(batch_iou):
        running_val_iou += batch_iou
        num_batches_with_valid_iou += 1

  avg_val_loss = running_val_loss / len(val_loader)
  avg_val_iou = running_val_iou / num_batches_with_valid_iou if num_batches_with_valid_iou > 0 else np.nan

  return avg_val_loss, avg_val_iou

def train_val(model, hparams, writer):
  # Device configuration
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  # hypar
  lr = hparams['learning_rate']
  bs = hparams['batch_size']
  opt_name = hparams['optimizer']

  model = model.to(device)
  loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

  if opt_name == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=lr)
  elif opt_name == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=lr)
  else:
    raise ValueError(f"Unknown optimizer: {opt_name}")

  # dataset
  train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, worker_init_fn=worker_init_fn)
  val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, worker_init_fn=worker_init_fn)

  # Early stopping
  best_val_loss = float('inf')
  patience_counter = 0
  patience = 2

  n_epoch = 10
  print(f"\nStarting training ...")
  for epoch in range(n_epoch):
    # --- Training Phase ---
    model, avg_train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
    writer.add_scalar('Loss/Train', avg_train_loss, epoch)

    # --- Validation Phase ---
    avg_val_loss,avg_val_iou = test_model(model, val_loader, loss_fn,
                                                device, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)

    writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
    writer.add_scalar('Metrics/Validation_IoU', avg_val_iou, epoch)

    if epoch % 1 == 0:
      print(f"Epoch {epoch+1}/{n_epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

    # --- early stop ---
    if avg_val_loss < best_val_loss:
      best_val_loss = avg_val_loss
      patience_counter = 0
    else:
      patience_counter += 1
      if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

  final_loss = avg_val_loss
  final_iou = avg_val_iou
  print("\nTraining complete.")
  return model, final_loss, avg_val_iou

def calculate_iou_mask(pred_mask, true_mask, num_classes, ignore_index):
  """
  Calculates Intersection over Union (IoU) for segmentation masks.
  Can handle binary or multi-class masks.
  """
  if pred_mask.shape != true_mask.shape:
    raise ValueError("Predicted mask and true mask must have the same shape.")

  # Ensure masks are integer types
  pred_mask = pred_mask.astype(np.int64)
  true_mask = true_mask.astype(np.int64)

  # Flatten masks for easier comparison
  pred_flat = pred_mask.flatten()
  true_flat = true_mask.flatten()

  if num_classes is None:
    all_classes = np.unique(np.concatenate((pred_flat, true_flat)))
    if ignore_index is not None:
        all_classes = all_classes[all_classes != ignore_index]
    num_classes = int(np.max(all_classes)) + 1 if len(all_classes) > 0 else 0
    if num_classes == 0:
        return 0.0 if ignore_index is None else np.nan

  iou_per_class = {}
  total_iou = 0.0
  valid_classes_count = 0

  for class_id in range(num_classes):
    if class_id == ignore_index:
      continue

    # Create binary masks for the current class
    pred_binary = (pred_flat == class_id)
    true_binary = (true_flat == class_id)

    intersection = np.sum(pred_binary & true_binary)
    union = np.sum(pred_binary | true_binary)

    if union == 0:
      iou_score = np.nan
    else:
      iou_score = intersection / union

    iou_per_class[class_id] = iou_score

    if not np.isnan(iou_score):
      total_iou += iou_score
      valid_classes_count += 1

  mean_iou = total_iou / valid_classes_count if valid_classes_count > 0 else np.nan

  return mean_iou, iou_per_class