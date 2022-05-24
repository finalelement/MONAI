import json
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
from monai.engines import SupervisedTrainer
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

from monai.apps.nuclick.dataset_prep import split_pannuke_dataset, split_nuclei_dataset
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
    json_path = os.path.normpath('/home/vishwesh/nuclick_experiments/try_1/data_list.json')

    groups = [
              "Neoplastic cells",
              "Inflammatory",
              "Connective/Soft tissue cells",
              "Dead Cells",
              "Epithelial",
        ]

    #Hyper-params
    patch_size = 128
    min_area = 5

    # Create Dataset
    if os.path.isfile(json_path) == 0:
        dataset_json = split_pannuke_dataset(image=img_data_path,
                                             label=label_data_path,
                                             output_dir=dataset_path,
                                             groups=groups)

        with open(json_path, 'w') as j_file:
            json.dump(dataset_json, j_file)
        j_file.close()
    else:
        with open(json_path, 'r') as j_file:
            dataset_json = json.load(j_file)
        j_file.close()

    ds_json_new = []
    for d in tqdm(dataset_json):
        ds_json_new.extend(split_nuclei_dataset(d, min_area=min_area))

    # Transforms
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
    data_set = Dataset(ds_json_new, transform=train_pre_transforms)
    train_data_loader = DataLoader(
                                   dataset=data_set,
                                   batch_size=32,
                                   shuffle=True,
                                   num_workers=2
                                )

    # Network Definition, Optimizer etc
    device = torch.device("cuda")

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
    #network.train()
    #TODO Refer here for how to fix up a validation when using a SupervisedTrainer. In short a supervisedevaluator needs to be created as a
    # training handler
    #TODO https://github.com/Project-MONAI/tutorials/blob/bc342633bd8e50be7b4a67b723006bb03285f6ba/acceleration/distributed_training/unet_training_workflows.py#L187
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=2,
        train_data_loader=train_data_loader,
        network=network,
        optimizer=optimizer,
        loss_function=dice_loss,
        inferer=SimpleInferer(),
        # if no FP16 support in GPU or PyTorch version < 1.6, will not enable AMP evaluation
        amp=False,
        postprocessing=train_post_transforms,
        #key_train_metric={"train_acc": Accuracy(output_transform=from_engine(["pred", "label"]), device=device)},
        #train_handlers=train_handlers,
    )
    trainer.run()

    '''
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
        print(f"Training ({step} / {len(train_data_loader)} Steps) (loss={loss})")
    '''

    print('Debug here')

    # End ...
    return None

if __name__=="__main__":
    main()