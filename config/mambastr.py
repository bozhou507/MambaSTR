train_batch_size: int=192
val_batch_size: int=192
test_batch_size: int=1024
num_train_workers: int=20
num_val_workers: int=20
num_test_workers: int=20

from .lmdb_dataset import *

model_name = None
train_workdir = f"work_dir/train_dir"
val_workdir = f"work_dir/val_dir"
test_workdir = f"work_dir/test_dir"

available_models = [
    'mambastr_tiny', 'mambastr_small', 'mambastr',
]
