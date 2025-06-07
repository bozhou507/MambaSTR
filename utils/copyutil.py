def copy_internal_nodes(o: any) -> any:
    """list and tuple are considered as internal nodes of the tree, and region types are considered as leaf nodes. This function only copies internal nodes, not leaf nodes."""
    if isinstance(o, list):
        return [copy_internal_nodes(i) for i in o]
    if isinstance(o, tuple):
        return tuple(copy_internal_nodes(i) for i in o)
    return o


if __name__ == '__main__':
    a = [[1, 2], (3, 4), ([5],)]
    b = copy_internal_nodes(a)
    assert b == a
    a[0][0] = 0
    a[1] = (-3, -4)
    a[2][0].append(3)
    assert a == [[0, 2], (-3, -4), ([5, 3],)]
    assert b == [[1, 2], (3, 4), ([5],)]