import torch
from torch import nn
from torchvision import transforms
from typing import Literal, Tuple
from utils.dataset import (RecogLMDBDataset, ImageFolderDataset)
from utils.dictionary import Dictionary, get_dictionary_by
from utils.transforms import CVGeometry, CVDeterioration, CVColorJitter
from .forward_loss import ForwardTextRecognitionLogits


def get_transform(expected_img_size: tuple, training: bool):
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
    return transforms.Compose(transform_list)


def get_image_folder_dataset(folder_path: str, expected_img_size: tuple, training=False):
    return ImageFolderDataset(
        folder_path=folder_path,
        transform=get_transform(expected_img_size, training)
    )


def get_ocr_lmdb_dataset(lmdb_path: str, expected_img_size: tuple, dictionary: Dictionary, training=False):
    return RecogLMDBDataset(
        lmdb_path=lmdb_path,
        transform=get_transform(expected_img_size, training),
        dictionary=dictionary
    )


class SceneTextRecognitionTask(nn.Module):

    def __init__(
        self,
        img_size: Tuple[int, int],
        dictionary: Dictionary=None,
        loss_type: Literal['ce', 'attn', 'ctc']=None,
        reduction: Literal['none', 'mean']='mean'):
        super().__init__()
        self.img_size = img_size
        if dictionary is None:
            self.dictionary = get_dictionary_by(loss_type)
        else:
            self.dictionary = dictionary
        self._postprocessor = ForwardTextRecognitionLogits(loss_type=loss_type, dictionary=self.dictionary, reduction=reduction)

    def get_image_folder_dataset(self, folder_path: str, training: bool):
        return get_image_folder_dataset(folder_path, self.img_size, training=training)

    def get_ocr_lmdb_dataset(self, lmdb_path: str, training: bool):
        return get_ocr_lmdb_dataset(lmdb_path, self.img_size, self.dictionary, training=training)

    def get_ocr_lmdb_datasets(self, lmdb_paths: str, training: bool):
        return [self.get_ocr_lmdb_dataset(lmdb_path, training) for lmdb_path in lmdb_paths]
    
    def postprocessor(self, logits: torch.Tensor, tgt: torch.Tensor=None, batch_weights: torch.Tensor=None):
        """
        
        Args:
            logits: [B, L, D]
            tgt: [B, S] if loss_type == 'ctc' else [B, L]
            batch_weights: [B]
        
        Returns:
            if tgt is None return predition
            if training return loss dict
            if testing return predition and ground truth
        """
        return self._postprocessor(logits, tgt, batch_weights)

    def forward(self, x: torch.Tensor, tgt: torch.Tensor=None, return_logits=False):
        """
        
        Args:
            x: [B, C, H, W]
            tgt: target
        """
        raise NotImplemented
