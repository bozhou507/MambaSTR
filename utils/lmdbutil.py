import os
import lmdb


def make_readonly_env(lmdb_path: str) -> lmdb.Environment:
    return lmdb.open(
        lmdb_path,
        max_readers=1,
        readonly=True,
        create=False,
        readahead=False,
        meminit=False,
        lock=False)


def is_lmdb(lmdb_path: str) -> bool:
    if not os.path.exists(lmdb_path):
        raise FileNotFoundError(f"\"{lmdb_path}\" is not a file or directory.")
    try:
        make_readonly_env(lmdb_path)
        return True
    except lmdb.Error:
        return False


def check_lmdb_integrity(lmdb_path: str) -> bool:
    env = make_readonly_env(lmdb_path)
    try:
        with env.begin(write=False) as txn:
            num_samples = int(txn.get(b"num-samples"))
            for i in range(num_samples):
                idx = f"{i+1:>09}"
                img_key = f'image-{idx}'
                label_key = f'label-{idx}'
                image_bin = txn.get(img_key.encode())
                label = txn.get(label_key.encode()).decode()  # 解码
                if label is None or image_bin is None:
                    return False
        return True
    except lmdb.CorruptedError:
        return False


def get_numsamples(lmdb_path: str) -> int:
    env = make_readonly_env(lmdb_path)
    assert env, f'Cannot open LMDB dataset from {lmdb_path}.'
    with env.begin(write=False) as txn:
        num_samples = int(txn.get(b"num-samples"))
    return num_samples
