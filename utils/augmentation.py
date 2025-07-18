class ContextAwareAugmenter:
  """
  A class for performing context-aware image augmentation.
  """
  def __init__(self, images, labels, isprs_colormap, isprs_classes):
    """
    Initializes the ContextAwareAugmenter with necessary data.
    """
    self.images = images
    self.labels = labels
    self.isprs_colormap = isprs_colormap
    self.isprs_classes = isprs_classes
    self.colormap2label = self._voc_colormap2label()
    self.car_pool = []
    self._crop_car_instances()

  def _voc_colormap2label(self):
    """Build a mapping from RGB to VOC category index (labels)"""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(self.isprs_colormap):
      colormap2label[
          (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

  def _voc_label_indices(self, colormap):
    """Map RGB values in VOC labels to their category indices"""
    if not isinstance(colormap, torch.Tensor):
          colormap = torch.tensor(colormap)

    if colormap.ndim == 3 and colormap.shape[0] != 3:
        colormap = colormap.permute(2, 0, 1)
    elif colormap.ndim == 2:
          if colormap.max() <= 255:
              colormap = colormap.unsqueeze(0).repeat(3, 1, 1)

    if colormap.ndim != 3 or colormap.shape[0] != 3:
          raise ValueError("Input colormap must be an RGB image tensor (C, H, W) with C=3")

    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
            + colormap[:, :, 2])
    return self.colormap2label[idx]


  def _crop_car_instances(self):
    """
    Crops and stores car instances from the labels and images.
    Populates the self.car_pool list with masked car instance tensors.
    """
    for i, label_tensor in enumerate(self.labels):
        original_image = self.images[i]
        label_indices = self._voc_label_indices(label_tensor)
        label_indices_np = label_indices.numpy()
        car_mask = (label_indices_np == 4).astype(np.uint8) * 255

        num_labels, labels_out, stats, centroids = cv2.connectedComponentsWithStats(car_mask, connectivity=8)

        for j in range(1, num_labels):
            left = stats[j, cv2.CC_STAT_LEFT]
            top = stats[j, cv2.CC_STAT_TOP]
            width = stats[j, cv2.CC_STAT_WIDTH]
            height = stats[j, cv2.CC_STAT_HEIGHT]
            area = stats[j, cv2.CC_STAT_AREA]

            instance_mask = (labels_out == j).astype(np.uint8) * 255

            x, y, w, h = cv2.boundingRect(instance_mask)

            img_height, img_width = original_image.shape[1], original_image.shape[2]
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_width - x)
            h = min(h, img_height - y)

            if w > 0 and h > 0:
                if isinstance(original_image, np.ndarray):
                    original_image = torch.tensor(original_image, dtype=torch.float32)
                    if original_image.ndim == 3 and original_image.shape[2] == 3:
                          original_image = original_image.permute(2, 0, 1)


                cropped_image_region = original_image[:, y:y+h, x:x+w]
                cropped_instance_mask = instance_mask[y:y+h, x:x+w]

                masked_car_instance = cropped_image_region * torch.tensor(cropped_instance_mask, dtype=torch.float32).unsqueeze(0).expand_as(cropped_image_region)/255.0

                self.car_pool.append(masked_car_instance)

    print(f"Successfully cropped and stored {len(self.car_pool)} car instances using masks.")

  def _find_valid_location(self, original_image, original_label_indices, selected_car_instance):
    """
    Finds a valid paste location for a car instance within the impervious surface polygon,
    avoiding overlap with existing cars.
    """
    car_height, car_width = selected_car_instance.shape[1], selected_car_instance.shape[2]
    img_height, img_width = original_image.shape[1], original_image.shape[2]
    imps_mask = (original_label_indices == 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(imps_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygon_contour = None
    if contours:
          largest_contour = max(contours, key=cv2.contourArea)
          epsilon = 3
          approx_polygon_impervious = cv2.approxPolyDP(largest_contour, epsilon=epsilon, closed=True)
          polygon_contour = approx_polygon_impervious
    else:
          return None, None, None


    max_attempts = 1000
    valid_location = None

    for attempt in range(max_attempts):
        if img_width - car_width < 0 or img_height - car_height < 0:
              return None, None, None

        random_x = random.randint(0, img_width - car_width)
        random_y = random.randint(0, img_height - car_height)

        points_to_check = []
        for r in np.linspace(random_y, random_y + car_height, 5, dtype=int):
            for c in np.linspace(random_x, random_x + car_width, 5, dtype=int):
                  r_checked = max(0, min(int(r), img_height - 1))
                  c_checked = max(0, min(int(c), img_width - 1))
                  points_to_check.append((c_checked, r_checked))

        all_points_inside_polygon = True
        if polygon_contour is not None and len(polygon_contour) > 0:
            for pt in points_to_check:
                if cv2.pointPolygonTest(polygon_contour, (int(pt[0]), int(pt[1])), measureDist=False) < 0:
                    all_points_inside_polygon = False
                    break
        else:
              all_points_inside_polygon = False
              break

        if all_points_inside_polygon:
            label_roi_np = original_label_indices[random_y:random_y+car_height, random_x:random_x+car_width]
            overlap_with_existing_car = np.any(label_roi_np == 4)

            if not overlap_with_existing_car:
                valid_location = (random_x, random_y)
                scaled_car_instance_np, car_mask_uint8 = self._prepare_car_instance(selected_car_instance)
                return valid_location, scaled_car_instance_np, car_mask_uint8

    return None, None, None

  def _prepare_car_instance(self, selected_car_instance):
    """
    Prepares the selected car instance and its mask for pasting.
    Converts to NumPy, scales to uint8, and creates a binary mask.
    """
    selected_car_instance_np = selected_car_instance.numpy()
    selected_car_instance_np = np.transpose(selected_car_instance_np, (1, 2, 0))

    car_mask_np = (selected_car_instance_np[:, :, 0] > 0).astype(np.uint8) * 255

    scaled_car_instance_np = np.zeros_like(selected_car_instance_np, dtype=np.uint8)
    non_zero_mask = selected_car_instance_np[:, :, 0] > 0

    min_val_car = np.min(selected_car_instance_np[non_zero_mask]) if np.any(non_zero_mask) else 0
    max_val_car = np.max(selected_car_instance_np[non_zero_mask]) if np.any(non_zero_mask) else 0

    if max_val_car > min_val_car:
        scaled_car_instance_np[non_zero_mask] = ((selected_car_instance_np[non_zero_mask] - min_val_car) / (max_val_car - min_val_car) * 255).astype(np.uint8)

    car_mask_uint8 = car_mask_np

    return scaled_car_instance_np, car_mask_uint8

  def _paste_car_on_image(self, original_image, scaled_car_instance_np, car_mask_uint8, valid_location):
    """
    Pastes the prepared car instance onto the original image at the specified location.
    """
    original_image_np = original_image.numpy()
    original_image_np = np.transpose(original_image_np, (1, 2, 0))

    original_image_uint8 = np.zeros_like(original_image_np, dtype=np.uint8)
    min_val_img = np.min(original_image_np)
    max_val_img = np.max(original_image_np)

    if max_val_img > min_val_img:
        original_image_uint8 = ((original_image_np - min_val_img) / (max_val_img - min_val_img) * 255).astype(np.uint8)
    else:
        original_image_uint8 = original_image_np.astype(np.uint8)

    car_height, car_width = scaled_car_instance_np.shape[0], scaled_car_instance_np.shape[1]

    x_offset, y_offset = valid_location
    roi = original_image_uint8[y_offset:y_offset+car_height, x_offset:x_offset+car_width]

    inverted_car_mask = cv2.bitwise_not(car_mask_uint8)

    car_mask_3_channel = cv2.cvtColor(car_mask_uint8, cv2.COLOR_GRAY2BGR)
    inverted_car_mask_3_channel = cv2.cvtColor(inverted_car_mask, cv2.COLOR_GRAY2BGR)

    car_foreground = cv2.bitwise_and(scaled_car_instance_np, car_mask_3_channel)
    background_roi = cv2.bitwise_and(roi, inverted_car_mask_3_channel)

    blended_roi = cv2.add(car_foreground, background_roi)

    augmented_image_np = original_image_uint8.copy()
    augmented_image_np[y_offset:y_offset+car_height, x_offset:x_offset+car_width] = blended_roi

    return augmented_image_np

  def _create_augmented_label(self, original_label_indices_np, car_mask_uint8, valid_location):
    """
    Generates a new label mask for the augmented image that includes the pasted car instance.
    """
    augmented_label_mask = original_label_indices_np.copy()

    x_offset, y_offset = valid_location
    car_height, car_width = car_mask_uint8.shape

    if y_offset + car_height > augmented_label_mask.shape[0] or x_offset + car_width > augmented_label_mask.shape[1]:
          print("Warning: Car instance goes beyond label mask bounds. Skipping label update.")
          return augmented_label_mask

    label_roi_augmented = augmented_label_mask[y_offset:y_offset+car_height, x_offset:x_offset+car_width]

    car_label_patch = np.zeros_like(label_roi_augmented, dtype=augmented_label_mask.dtype)
    car_label_patch[car_mask_uint8 > 0] = 4

    augmented_label_mask[y_offset:y_offset+car_height, x_offset:x_offset+car_width][car_mask_uint8 > 0] = car_label_patch[car_mask_uint8 > 0]

    return augmented_label_mask.astype(np.int64)


  def augment_image(self, image_index):
    """
    Orchestrates the process of selecting a car, finding a location, preparing the instance,
    pasting it, and generating the new label for a given image index.
    """
    if image_index < 0 or image_index >= len(self.images):
        print(f"Error: Image index {image_index} is out of bounds.")
        return None, None

    original_image = self.images[image_index]
    original_label = self.labels[image_index]

    original_label_indices_np = self._voc_label_indices(original_label).numpy()

    if not self.car_pool:
        print(f"Error: Car pool is empty. Cannot augment image {image_index}.")
        return None, None
    selected_car_instance = random.choice(self.car_pool)

    valid_location, scaled_car_instance_np, car_mask_uint8 = self._find_valid_location(
        original_image, original_label_indices_np, selected_car_instance
    )

    if valid_location is None:
        return None, None

    augmented_image_np = self._paste_car_on_image(
        original_image, scaled_car_instance_np, car_mask_uint8, valid_location
    )

    augmented_label_mask_np = self._create_augmented_label(
        original_label_indices_np, car_mask_uint8, valid_location
    )

    augmented_image_tensor = torch.tensor(augmented_image_np, dtype=torch.float32).permute(2, 0, 1)

    augmented_label_mask_tensor = torch.tensor(augmented_label_mask_np, dtype=torch.long)

    print(f"augment image {image_index}.")
    return augmented_image_tensor, augmented_label_mask_tensor