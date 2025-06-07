from typing import List, Iterable
from torchvision import transforms
from copy import deepcopy
import pickle
import itertools
import numpy as np

from utils.dataset import (RecogLMDBDataset, ImageFolderDataset)
from utils.dictionary import Dictionary
from utils.transforms import CVGeometry, CVDeterioration, CVColorJitter


def get_image_folder_dataset(
    folder_path: str,
    expected_img_size: tuple,
    with_raw_image=False,
    training: bool=False):
    transform_list = []
    if training:
        transform_list.extend([
            CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
            CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
            CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25),
        ])
    transform_list.extend([
        transforms.ToTensor(),  # [0, 255] -> [0.0, 1.0]
        transforms.Resize(size=expected_img_size, antialias=True),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])
    return ImageFolderDataset(
        folder_path=folder_path,
        transform=transforms.Compose(transform_list),
        with_raw_image=with_raw_image
    )


def get_ocr_lmdb_dataset(
    lmdb_path: str,
    expected_img_size: tuple,
    dictionary: Dictionary,
    with_raw_image=False,
    training: bool=False):
    transform_list = []
    if training:
        transform_list.extend([
            CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
            CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
            CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25),
        ])
    transform_list.extend([
        transforms.ToTensor(),  # [0, 255] -> [0.0, 1.0]
        transforms.Resize(size=expected_img_size, antialias=True),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])
    return RecogLMDBDataset(
        lmdb_path=lmdb_path,
        transform=transforms.Compose(transform_list),
        dictionary=dictionary,
        with_raw_image=with_raw_image
    )
