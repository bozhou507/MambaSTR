import re
import io
from PIL import Image
from PIL import ImageFile
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from typing import Literal
from .dictionary import Dictionary
from . import osutil
from . import lmdbutil
from .collate_fn import default_collate_fn
from abc import ABC, abstractmethod
from typing import List

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(ABC, Dataset):

    @abstractmethod
    def __len__(self):
        pass

    @staticmethod
    def collate_fn(batch):
        return default_collate_fn(batch)


class ConcatTrainDataset(ConcatDataset, BaseDataset):
    pass

class ConcatValDataset(ConcatDataset, BaseDataset):
    def __init__(self, datasets, dataset_names: List[str]=None) -> None:
        super().__init__(datasets)
        assert len(datasets) == len(dataset_names)
        self.metainfo = {
            "cumulative_sizes": self.cumulative_sizes,
            "dataset_names": dataset_names
        }

ConcatTestDataset = ConcatValDataset


class RecogLMDBDataset(BaseDataset):

    def __init__(
        self,
        lmdb_path: str,
        dictionary: Dictionary,
        transform: transforms.Compose=None,
        mode: Literal["RGB", "L"]="RGB",
        with_raw_image: bool=False,  # works when transform is not None
        with_target: bool=True
    ):
        self.dictionary = dictionary
        self.transform = transform
        self.mode = mode
        self.lmdb_path = lmdb_path
        if with_raw_image and self.transform is None:
            import warnings
            warnings.warn('transform is None, with_raw_image=True will be ignored')
            with_raw_image = False
        self.with_raw_image = with_raw_image
        self.with_target = with_target
        self.length = lmdbutil.get_numsamples(self.lmdb_path)
        # do not open lmdb here!!
        # https://github.com/pytorch/vision/issues/689#issuecomment-787215916

    def open_lmdb(self):
        self.env = lmdbutil.make_readonly_env(self.lmdb_path)
        assert self.env, f'Cannot open LMDB dataset from {self.lmdb_path}.'
    
    def close_lmdb(self):
        if self.env is not None:
            del self.env

    def __len__(self):
        return self.length

    def get(self, idx):
        with self.env.begin(write=False) as txn:
            image_key, label_key = f'image-{idx+1:09d}', f'label-{idx+1:09d}'
            label = str(txn.get(label_key.encode()), 'utf-8')  # label
            label = re.sub('[^0-9a-zA-Z]+', '', label)

            imgbuf = txn.get(image_key.encode())  # image
            image = Image.open(io.BytesIO(imgbuf)).convert(self.mode)
            return image, label

    def __getitem__(self, idx):
        if not hasattr(self, 'env'):
            self.open_lmdb()

        image, label = self.get(idx)

        if self.with_raw_image:
            return_items = [image]
        else:
            return_items = []

        if self.transform is not None:
            image = self.transform(image)

        if isinstance(image, tuple):
            return_items.extend(image)
        else:
            return_items.append(image)

        if self.with_target:
            target = self.dictionary.word2idx(label)
            target = torch.tensor(target)
            return_items.append(target)

        if len(return_items) == 1:
            return return_items[0]
        return tuple(return_items)


class ImageFolderDataset(BaseDataset):

    def __init__(
        self,
        folder_path: str,
        transform: transforms.Compose=None,
        mode: Literal["RGB", "L"]="RGB",
        with_raw_image: bool=False,
        with_img_path: bool=True
    ):
        self.transform = transform
        self.mode = mode
        self.image_paths = sorted(osutil.list_files(folder_path))
        if with_raw_image and self.transform is None:
            import warnings
            warnings.warn('transform is None, with_raw_image=True will be ignored')
            with_raw_image = False
        self.with_raw_image = with_raw_image
        self.with_img_path = with_img_path
        self.length = len(self.image_paths)

    def __len__(self):
        return self.length

    def get(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert(self.mode)
        image_name = osutil.basename(image_path)
        return image, image_name

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert(self.mode)

        if self.with_raw_image:
            return_items = [image]
        else:
            return_items = []

        if self.transform is not None:
            image = self.transform(image)

        if isinstance(image, tuple):
            return_items.extend(image)
        else:
            return_items.append(image)

        if self.with_img_path:
            return_items.append(image_path)

        if len(return_items) == 1:
            return return_items[0]
        return tuple(return_items)
