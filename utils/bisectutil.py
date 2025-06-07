"""Bisection algorithms.

Modified from <https://github.com/python/cpython/blob/main/Lib/bisect.py>
"""
from typing import Sequence, Callable, Optional


def bisect_right(
    sorted_data: Sequence,
    target: any,
    lo: int=0,
    hi: Optional[int]=None,
    *,
    key: Optional[Callable]=None,
    reversed_order: bool=False
):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(i, x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(sorted_data)
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    if reversed_order is False:
        if key is None:
            while lo < hi:
                mid = (lo + hi) // 2
                if target < sorted_data[mid]:
                    hi = mid
                else:
                    lo = mid + 1
        else:
            while lo < hi:
                mid = (lo + hi) // 2
                if target < key(sorted_data[mid]):
                    hi = mid
                else:
                    lo = mid + 1
    else:
        if key is None:
            while lo < hi:
                mid = (lo + hi) // 2
                if sorted_data[mid] < target:
                    hi = mid
                else:
                    lo = mid + 1
        else:
            while lo < hi:
                mid = (lo + hi) // 2
                if key(sorted_data[mid]) < target:
                    hi = mid
                else:
                    lo = mid + 1
    return lo


def bisect_left(
    sorted_data: Sequence,
    target: any,
    lo: int=0,
    hi: Optional[int]=None,
    *,
    key: Optional[Callable]=None,
    reversed_order: bool=False
):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(i, x) will
    insert just before the leftmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(sorted_data)
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    if reversed_order is False:
        if key is None:
            while lo < hi:
                mid = (lo + hi) // 2
                if sorted_data[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
        else:
            while lo < hi:
                mid = (lo + hi) // 2
                if key(sorted_data[mid]) < target:
                    lo = mid + 1
                else:
                    hi = mid
    else:
        if key is None:
            while lo < hi:
                mid = (lo + hi) // 2
                if target < sorted_data[mid]:
                    lo = mid + 1
                else:
                    hi = mid
        else:
            while lo < hi:
                mid = (lo + hi) // 2
                if target < key(sorted_data[mid]):
                    lo = mid + 1
                else:
                    hi = mid
    return lo
