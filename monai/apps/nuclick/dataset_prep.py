import os
import cv2
import math
import copy
from tqdm import tqdm
import numpy as np
from skimage.measure import regionprops
from monai.transforms import LoadImage

def split_pannuke_dataset(image, label, output_dir, groups):
    groups = groups if groups else dict()
    groups = [groups] if isinstance(groups, str) else groups
    if not isinstance(groups, dict):
        groups = {v: k + 1 for k, v in enumerate(groups)}

    label_channels = {
        0: "Neoplastic cells",
        1: "Inflammatory",
        2: "Connective/Soft tissue cells",
        3: "Dead Cells",
        4: "Epithelial",
    }

    print(f"++ Using Groups: {groups}")
    print(f"++ Using Label Channels: {label_channels}")
    #logger.info(f"++ Using Groups: {groups}")
    #logger.info(f"++ Using Label Channels: {label_channels}")

    images = np.load(image)
    labels = np.load(label)
    print(f"Image Shape: {images.shape}")
    print(f"Labels Shape: {labels.shape}")
    #logger.info(f"Image Shape: {images.shape}")
    #logger.info(f"Labels Shape: {labels.shape}")

    images_dir = output_dir
    labels_dir = os.path.join(output_dir, "labels", "final")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    dataset_json = []
    for i in tqdm(range(images.shape[0])):
        name = f"img_{str(i).zfill(4)}.npy"
        image_file = os.path.join(images_dir, name)
        label_file = os.path.join(labels_dir, name)

        image_np = images[i]
        mask = labels[i]
        label_np = np.zeros(shape=mask.shape[:2])

        for idx, name in label_channels.items():
            if idx < mask.shape[2]:
                m = mask[:, :, idx]
                if np.count_nonzero(m):
                    m[m > 0] = groups.get(name, 1)
                    label_np = np.where(m > 0, m, label_np)

        np.save(image_file, image_np)
        np.save(label_file, label_np)
        dataset_json.append({"image": image_file, "label": label_file})

    return dataset_json

def split_nuclei_dataset(d, centroid_key="centroid", mask_value_key="mask_value", min_area=5):
    dataset_json = []

    mask = LoadImage(image_only=True, dtype=np.uint8)(d["label"])
    _, labels, _, _ = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

    stats = regionprops(labels)
    for stat in stats:
        if stat.area < min_area:
            #logger.debug(f"++++ Ignored label with smaller area => ( {stat.area} < {min_area})")
            print(f"++++ Ignored label with smaller area => ( {stat.area} < {min_area})")
            continue

        x, y = stat.centroid
        x = int(math.floor(x))
        y = int(math.floor(y))

        item = copy.deepcopy(d)
        item[centroid_key] = (x, y)
        item[mask_value_key] = stat.label

        # logger.info(f"{d['label']} => {len(stats)} => {mask.shape} => {stat.label}")
        dataset_json.append(item)
    return dataset_json