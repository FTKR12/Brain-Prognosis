import os
import csv
import glob

from monai.data import (
    ImageDataset,
    DataLoader,
)
from monai.transforms import(
    Compose,
    ScaleIntensity,
    EnsureChannelFirst,
    Resize,
    RandRotate90
)

def build_dataloader(args):

    # load data
    train_paths, val_paths, test_paths = [], [], []
    train_id = glob.glob(f"{args.dataset_dir}/{args.synthesize_model}/train/{args.input}/*")
    val_id = glob.glob(f"{args.dataset_dir}/{args.synthesize_model}/val/{args.input}/*")
    test_id = glob.glob(f"{args.dataset_dir}/{args.synthesize_model}/test/{args.input}/*")
    with open(args.ans_path, "r", encoding='utf-8') as f:
        info = csv.reader(f, delimiter=",", doublequote=True, lineterminator="\n", quotechar='"', skipinitialspace=True)
        header = next(info)
        prog_dict = {int(x[0])-1001: x[41] for x in info}
    
    train_img_paths, train_ans_paths = [], []
    val_img_paths, val_ans_paths = [], []
    test_img_paths, test_ans_paths = [], []
    for data_id in train_id:
        if int(data_id.split("/")[-1].replace(".nii.gz", "")) < 60:
            train_img_paths.append(data_id)
            train_ans_paths.append(int(prog_dict[int(data_id.split('/')[-1].replace('.nii.gz',''))]))
    for data_id in val_id:
        if int(data_id.split("/")[-1].replace(".nii.gz", "")) < 60:
            val_img_paths.append(data_id)
            val_ans_paths.append(int(prog_dict[int(data_id.split('/')[-1].replace('.nii.gz',''))]))
    for data_id in test_id:
        if int(data_id.split("/")[-1].replace(".nii.gz", "")) < 60:
            test_img_paths.append(data_id)
            test_ans_paths.append(int(prog_dict[int(data_id.split('/')[-1].replace('.nii.gz',''))]))

    # augmentation
    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()])
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])

    # maek dataset
    train_ds = ImageDataset(image_files=train_img_paths, transform=train_transforms, labels=train_ans_paths)
    val_ds = ImageDataset(image_files=val_img_paths, transform=val_transforms, labels=val_ans_paths)
    test_ds = ImageDataset(image_files=test_img_paths, transform=val_transforms, labels=test_ans_paths)
    
    # make dataloader
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.test_batch_size, num_workers=4)

    return train_loader, val_loader, test_loader