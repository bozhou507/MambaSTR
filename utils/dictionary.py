# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Literal


def list_from_file(filename, encoding='utf-8'):
    item_list = []
    with open(filename, encoding=encoding) as f:
        for line in f:
            item_list.append(line.rstrip('\n\r'))
    return item_list


class Dictionary:
    def __init__(self,
                 dict_file: str,
                 max_word_length: int = 0,
                 with_start: bool = False,
                 with_end: bool = False,
                 same_start_end: bool = False,
                 with_padding: bool = False,
                 with_unknown: bool = False,
                 with_mask: bool = False,
                 start_token: str = '<BOS>',
                 end_token: str = '<EOS>',
                 start_end_token: str = '<BOS/EOS>',
                 padding_token: str = '<PAD>',
                 unknown_token: str = '<UKN>',
                 mask_token = '<MSK>',
                 ignore_case: bool=False) -> None:
        self.max_word_length = max_word_length
        self.with_start = with_start
        self.with_end = with_end
        self.same_start_end = same_start_end
        self.with_padding = with_padding
        self.with_unknown = with_unknown
        self.with_mask = with_mask
        self.start_end_token = start_end_token
        self.start_token = start_token
        self.end_token = end_token
        self.padding_token = padding_token
        self.unknown_token = unknown_token
        self.mask_token = mask_token
        self.ignore_case = ignore_case
        self.special_indexes = []

        assert isinstance(dict_file, str)
        self._dict = []
        for line_num, line in enumerate(list_from_file(dict_file)):
            line = line.strip('\r\n')
            if len(line) > 1:
                raise ValueError('Expect each line has 0 or 1 character, '
                                 f'got {len(line)} characters '
                                 f'at line {line_num + 1}')
            if line != '':
                self._dict.append(line)

        self._update_dict()
        self._char2idx = {char: idx for idx, char in enumerate(self._dict)}
        
        assert len(set(self._dict)) == len(self._dict), \
            'Invalid dictionary: Has duplicated characters.'

    def set_max_word_length(self, max_word_length: int):
        self.max_word_length = max_word_length

    @property
    def num_classes(self) -> int:
        """int: Number of output classes. Special tokens are counted.
        """
        return len(self._dict)

    @property
    def dict(self) -> list:
        """list: Returns a list of characters to recognize, where special
        tokens are counted."""
        return self._dict

    def char2idx(self, char: str, strict: bool = True) -> int:
        """Convert a character to an index via ``Dictionary.dict``.

        Args:
            char (str): The character to convert to index.
            strict (bool): The flag to control whether to raise an exception
                when the character is not in the dictionary. Defaults to True.

        Return:
            int: The index of the character.
        """
        if self.ignore_case:
            char = char.lower()
        char_idx = self._char2idx.get(char, None)
        if char_idx is None:
            if self.with_unknown:
                return self.unknown_idx
            elif not strict:
                return None
            else:
                raise Exception(f'Chararcter: {char} not in dict,'
                                ' please check gt_label and use'
                                ' custom dict file,'
                                ' or set "with_unknown=True"')
        return char_idx

    def word2idx(self, string: str) -> List:
        """Convert a string to a list of indexes via ``Dictionary.dict``.

        Args:
            string (str): The string to convert to indexes.

        Return:
            list: The list of indexes of the string.
        """
        idx = list()
        if self.with_start:
            idx.append(self.start_idx)
        for s in string:
            char_idx = self.char2idx(s)
            if char_idx is None:
                if self.with_unknown:
                    continue
                raise Exception(f'Chararcter: {s} not in dict,'
                                ' please check gt_label and use'
                                ' custom dict file,'
                                ' or set "with_unknown=True"')
            idx.append(char_idx)
        if self.with_end:
            idx.append(self.end_idx)
        if self.with_padding:
            if len(idx) < self.max_word_length:
                idx.extend([self.padding_idx] * (self.max_word_length - len(idx)))
        if self.max_word_length > 0:
            idx = idx[:self.max_word_length]
        return idx

    def idx2str(self, index: Sequence[int], ignore_indexes: List[int] = [], break_at_end_idx: bool=True) -> str:
        """Convert a list of index to string.

        Args:
            index (list[int]): The list of indexes to convert to string.

        Return:
            str: The converted string.
        """
        # assert isinstance(index, (list, tuple))
        string = ''
        for i in index:
            assert i < len(self._dict), f'Index: {i} out of range! Index ' \
                                        f'must be less than {len(self._dict)}'
            if i == self.end_idx and break_at_end_idx:
                break
            if i not in ignore_indexes:
                string += self._dict[i]
        return string

    def _update_dict(self):
        """Update the dict with tokens according to parameters."""
        # BOS/EOS
        self.start_idx = None
        self.end_idx = None
        if self.with_start and self.with_end and self.same_start_end:
            self._dict.append(self.start_end_token)
            self.start_idx = len(self._dict) - 1
            self.end_idx = self.start_idx
            self.special_indexes.append(self.start_idx)
        else:
            if self.with_start:
                self._dict.append(self.start_token)
                self.start_idx = len(self._dict) - 1
                self.special_indexes.append(self.start_idx)
            if self.with_end:
                self._dict.append(self.end_token)
                self.end_idx = len(self._dict) - 1
                self.special_indexes.append(self.end_idx)

        # padding
        self.padding_idx = None
        if self.with_padding:
            self._dict.append(self.padding_token)
            self.padding_idx = len(self._dict) - 1
            self.special_indexes.append(self.padding_idx)

        # unknown
        self.unknown_idx = None
        if self.with_unknown and self.unknown_token is not None:
            self._dict.append(self.unknown_token)
            self.unknown_idx = len(self._dict) - 1
            self.special_indexes.append(self.unknown_idx)

        # MASK TOKEN
        self.mask_idx = None
        if self.with_mask and self.mask_token is not None:
            self._dict.append(self.mask_token)
            self.mask_idx = len(self.dict) - 1
            self.special_indexes.append(self.unknown_idx)


def get_dictionary(dict_file) -> Dictionary:
    return Dictionary(
        dict_file=dict_file,
        max_word_length=36,
        with_start=True,
        with_end=True,
        with_unknown=True,
        with_padding=True,
        ignore_case=True
    )


dictionary = get_dictionary('dicts/lower_english_digits.txt')  # 36 + 4


def get_dictionary_for_ctc(
    dict_file: str='dicts/lower_english_digits.txt',
    max_word_length: int=25) -> Dictionary:
    """with padding, unknown, ignore cases"""
    return Dictionary(
        dict_file=dict_file,
        with_padding=True,
        with_unknown=True,
        ignore_case=True,
        max_word_length=max_word_length)


def get_dictionary_for_ce(
    dict_file: str='dicts/lower_english_digits.txt',
    max_word_length: int=25) -> Dictionary:
    """with end, padding, unknown, ignore cases"""
    return Dictionary(
        dict_file=dict_file,
        with_end=True,
        with_padding=True,
        with_unknown=True,
        ignore_case=True,
        max_word_length=max_word_length)


def get_dictionary_for_attn(
    dict_file: str='dicts/lower_english_digits.txt',
    max_word_length: int=26,
    same_start_end: bool=False) -> Dictionary:
    """with start, end, padding, unknown, ignore cases"""
    return Dictionary(
        dict_file=dict_file,
        with_start=True,
        with_end=True,
        same_start_end=same_start_end,
        with_padding=True,
        with_unknown=True,
        ignore_case=True,
        max_word_length=max_word_length)


def get_dictionary_by(loss_type: Literal['ce', 'attn', 'ctc'], **kwargs) -> Dictionary:
    try:
        import sys
        return getattr(sys.modules[__name__], f"get_dictionary_for_{loss_type}")(**kwargs)
    except:
        raise RuntimeError(f"Unsupported loss_type {loss_type}")


if __name__ == '__main__':
    print(dictionary.dict)
    print(dictionary._char2idx)
    print(dictionary.start_idx, dictionary.start_token)
    print(dictionary.end_idx, dictionary.end_token)
    print(dictionary.unknown_idx, dictionary.unknown_token)
    print(dictionary.word2idx("ABCabc"))
