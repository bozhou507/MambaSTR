import os
import shutil
from typing import List, Literal


def abspath(path: str) -> str:
    """Return an absolute path."""
    return os.path.abspath(path)


def relpath(path: str, root_dir: str) -> str:
    """Return an relative path with respect to root_dir."""
    return os.path.relpath(path, root_dir)


def dirpath(path: str) -> str:
    """Return an absolute path represent the dirpath of the input path."""
    return os.path.dirname(abspath(path))


def parentpath(dirpath: str) -> str:
    """Return an absolute path of the parent directory of the input dirpath"""
    return abspath(os.path.join(dirpath, os.pardir))


def basename(path: str) -> str:
    """Returns the final component of a path."""
    return os.path.basename(path)


def replace_extension(path: str, suffix: str) -> str:
    """Replace extension in path with suffix.

    Examples:
        1: replace_extension("a.txt", ".csv") -> "a.csv"
        2: replace_extension("a_txt", ".csv") -> "a_txt.csv"
    """
    root, _ = os.path.splitext(path)
    return root + suffix


def isfile(path: str) -> bool:
    """Test whether a path is a regular file."""
    return os.path.isfile(path)


def isdir(path: str) -> bool:
    """Return true if the pathname refers to an existing directory."""
    return os.path.isdir(path)


def islink(path: str) -> bool:
    """Test whether a path is a symbolic link."""
    return os.path.islink(path)


def is_path_exist(path: str) -> bool:
    """Test whether a path exists.  Returns False for broken symbolic links"""
    return os.path.exists(path)


def get_info(path: str) -> dict:
    return {
        'size': os.path.getsize(path),
        'created_time': os.path.getctime(path),
        'modified_time': os.path.getmtime(path)
    }


def concat_paths(paths: List[str]) -> str:
    return os.path.join(*paths)


def _list_paths(
        root_dir: str,
        recursion: bool=False,
        follow_links=False,
        item_type: Literal['all', 'file', 'dir']='all',
        leaf_only=True) -> List[str]:
    """get files or directories in root_dir
    
    Args:
        root_dir: pass
        recursion: whether walk the directories recursively or not
        item_type: pass
        leaf_only: whether include paths which is a prefix of a leaf or not, active when recursion is True

    Returns:
        A list of paths.

    Raises:
        pass
    """
    file_only = item_type == 'file'
    dir_only = item_type == 'dir'
    path_list = []
    if not recursion:
        for name in os.listdir(root_dir):
            path = concat_paths([root_dir, name])
            if file_only and isdir(path) or dir_only and isfile(path):
                continue
            path_list.append(path)
        return path_list

    for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=follow_links):
        if file_only:
            for filename in filenames:
                filepath = concat_paths([dirpath, filename])
                path_list.append(filepath)
        elif dir_only:
            if not leaf_only or not dirnames:
                path_list.append(dirpath)
        else:
            if not leaf_only or not dirnames and not filenames:
                path_list.append(dirpath)
            for filename in filenames:
                filepath = concat_paths([dirpath, filename])
                path_list.append(filepath)
    if path_list and path_list[0] == root_dir:
        path_list.pop(0)
    return path_list


def list_files(path: str, recursion: bool=False, follow_links=False, suffix: str='') -> List[str]:
    """list files' path under the `path` folder"""
    paths = _list_paths(path, item_type='file', recursion=recursion, follow_links=follow_links)
    if suffix:
        paths = list(filter(lambda f: f.endswith(suffix), paths))
    return paths


def list_dirs(path: str, recursion: bool=False, follow_links=False, leaf_only=True) -> List[str]:
    """list dirs' path under the `path` folder"""
    return _list_paths(path, item_type='dir', recursion=recursion, follow_links=follow_links, leaf_only=leaf_only)


def list_all(path: str, recursion: bool=False, follow_links=False, leaf_only=True) -> List[str]:
    """list files and dirs' path under the `path` folder"""
    return _list_paths(path, item_type='all', recursion=recursion, follow_links=follow_links, leaf_only=leaf_only)


def create(path: str, is_directory: bool=False):
    if is_directory:
        os.makedirs(path, exist_ok=True)
    else:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        if not is_path_exist(path):
            open(path, 'w').close()


def create_file(file_path: str):
    create(file_path, False)


def create_dir(dir_path: str):
    create(dir_path, True)


def create_file_with_size(filepath: str, num_bytes, byte=b'\0'):
    if is_path_exist(filepath):
        raise FileExistsError(f'Path `{filepath}` already exists.')
    with open(filepath, 'wb') as f:
        f.write(byte * num_bytes)


make_file = create_file
make_dir = create_dir
make_file_with_size = create_file_with_size


def split_file_to_pieces(filepath: str, num_pieces: int, with_order=False, order_num_bytes=4, byteorder='little'):
    with open(filepath, 'rb') as f:
        all_bytes = f.read()
    total_num_bytes = len(all_bytes)
    if total_num_bytes < num_pieces:
        raise RuntimeError(f'Impossible to split {total_num_bytes} bytes to {num_pieces} pieces.')
    piece_num_bytes = total_num_bytes // num_pieces
    remainder = piece_num_bytes % num_pieces
    start = 0
    for i in range(num_pieces):
        current_num_bytes  = piece_num_bytes + (i < remainder)
        with open(f'{filepath}_{i}.piece', 'wb') as f:
            part_bytes = all_bytes[start: start + current_num_bytes]
            if with_order:
                part_bytes = i.to_bytes(order_num_bytes, byteorder) + part_bytes
            f.write(part_bytes)
        start += current_num_bytes
    remove_file(filepath)


def combine_file_from_pieces(filepath: str, with_order=False, order_num_bytes=4, byteorder='little'):
    if is_path_exist(filepath):
        raise FileExistsError
    with open(filepath, 'wb') as f:
        i = -1
        while True:
            i += 1
            piece_file_path = f"{filepath}_{i}.piece"
            if not is_path_exist(piece_file_path):
                break
            with open(piece_file_path, 'rb') as f2:
                part_bytes = f2.read()
                if with_order:
                    assert int.from_bytes(part_bytes[:order_num_bytes], byteorder) == i
                    part_bytes = part_bytes[order_num_bytes:]
                f.write(part_bytes)
            remove_file(piece_file_path)


def move(source: str, destination: str):
    shutil.move(source, destination)


def copy(source: str, destination: str, is_directory: bool=False):
    if is_directory:
        shutil.copytree(source, destination)
    else:
        assert not isdir(source)
        shutil.copy2(source, destination)


def remove(path: str, is_directory: bool=False):
    if is_directory:
        shutil.rmtree(path)
    else:
        assert not isdir(path)
        os.remove(path)


def remove_file(filepath: str):
    remove(filepath, False)


def remove_dir(dirpath: str):
    remove(dirpath, True)


delete = remove
delete_file = remove_file
delete_dir = remove_dir


from pathlib import Path
from itertools import islice


def tree(dir_path: Path, level: int=-1, only_directories: bool=False,
         length_limit: int=1000):
    """
    Changed from https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python/59109706#59109706

    Given a directory Path object print a visual tree structure
    """
    space =  '    '
    branch = '│   '
    tee =    '├── '
    last =   '└── '
    dir_path = Path(dir_path) # accept string coerceable to Path
    files = 0
    directories = 0
    def inner(dir_path: Path, prefix: str='', level=-1):
        nonlocal files, directories
        if not level: 
            return # 0, stop iterating
        if only_directories:
            contents = [d for d in dir_path.iterdir() if d.is_dir()]
        else: 
            contents = list(dir_path.iterdir())
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            if path.is_dir():
                yield prefix + pointer + path.name + '/'
                directories += 1
                extension = branch if pointer == tee else space 
                yield from inner(path, prefix=prefix+extension, level=level-1)
            elif not only_directories:
                yield prefix + pointer + path.name
                files += 1
    print(dir_path.name)
    iterator = inner(dir_path, level=level)
    for line in islice(iterator, length_limit):
        print(line)
    if next(iterator, None):
        print(f'... length_limit, {length_limit}, reached, counted:')
    print(f'\n{directories} directories' + (f', {files} files' if files else ''))
