import logging
import math
import os
import random
import cv2
import numpy as np
import skimage
import torch
from tqdm import tqdm
#from lib.handlers import TensorBoardImageHandler
#from lib.transforms import FilterImaged
#from lib.utils import split_dataset, split_nuclei_dataset
from monai.config import KeysCollection
from monai.handlers import MeanDice, from_engine
from monai.inferers import SimpleInferer
from monai.losses import DiceLoss
from monai.networks.nets import BasicUNet
from monai.data import (
    CacheDataset,
    Dataset,
    DataLoader,
)
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsChannelFirstd,
    AsDiscreted,
    Compose,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    RandomizableTransform,
    RandRotate90d,
    ScaleIntensityRangeD,
    ToNumpyd,
    TorchVisiond,
    ToTensord,
    Transform,
)

from monai.apps.nuclick.dataset_prep import split_pannuke_dataset
from monai.apps.nuclick.transforms import (
    FlattenLabeld,
    ExtractPatchd,
    SplitLabeld,
    AddPointGuidanceSignald,
    FilterImaged
)

#from monailabel.interfaces.datastore import Datastore
#from monailabel.tasks.train.basic_train import BasicTrainTask, Context

def main():

    # Paths
    img_data_path = os.path.normpath('/scratch/pan_nuke_data/fold_1/Fold_1/images/fold1/images.npy')
    label_data_path = os.path.normpath('/scratch/pan_nuke_data/fold_1/Fold_1/masks/fold1/masks.npy')
    dataset_path = os.path.normpath('/home/vishwesh/nuclick_experiments/try_1/data')

    groups = [
              "Neoplastic cells",
              "Inflammatory",
              "Connective/Soft tissue cells",
              "Dead Cells",
              "Epithelial",
        ]

    # Create Dataset
    dataset_json = split_pannuke_dataset(image=img_data_path,
                                         label=label_data_path,
                                         output_dir=dataset_path,
                                         groups=groups)

    # Transforms
    patch_size = 128
    min_area = 5
    train_pre_transforms = Compose(
        [
            LoadImaged(keys=("image", "label"), dtype=np.uint8),
            FilterImaged(keys="image", min_size=5),
            FlattenLabeld(keys="label"),
            AsChannelFirstd(keys="image"),
            AddChanneld(keys="label"),
            ExtractPatchd(keys=("image", "label"), patch_size=patch_size),
            SplitLabeld(label="label", others="others", mask_value="mask_value", min_area=min_area),
            ToTensord(keys="image"),
            TorchVisiond(
                keys="image", name="ColorJitter", brightness=64.0 / 255.0, contrast=0.75, saturation=0.25, hue=0.04
            ),
            ToNumpyd(keys="image"),
            RandRotate90d(keys=("image", "label", "others"), prob=0.5, spatial_axes=(0, 1)),
            ScaleIntensityRangeD(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
            AddPointGuidanceSignald(image="image", label="label", others="others"),
            EnsureTyped(keys=("image", "label"))
        ]
    )

    train_post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold_values=True, logit_thresh=0.5),
        ]
    )


    # Define Dataset & Loading
    data_set = Dataset(dataset_json, transform=train_pre_transforms)
    train_data_loader = DataLoader(
                                   dataset=data_set,
                                   batch_size=8,
                                   shuffle=True,
                                   num_workers=2
                                )

    # Network Definition, Optimizer etc
    device = 'cuda'

    network = BasicUNet(
        spatial_dims=2,
        in_channels=5,
        out_channels=1,
        features=(32, 64, 128, 256, 512, 32),
    )

    network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), 0.0001)
    dice_loss = DiceLoss(sigmoid=True, squared_pred=True)

    # Training Process
    #TODO Consider uisng the Supervised Trainer over here from MONAI
    network.train()

    epoch_loss = 0
    for step, batch in enumerate(train_data_loader):
        step += 1
        x, y = (batch["image"].to(device), batch["label"].to(device))
        logit_map = network(x)
        loss = dice_loss(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        print("Training ({0:%d} / {1:%d} Steps) (loss={2:%2.5f})".format(step, len(train_data_loader), loss))


    print('Debug here')

    # End ...
    return None

if __name__=="__main__":
    main()