def sparse_dia_memory_usage(mat):
    try:
        return mat.data.nbytes + mat.offsets.nbytes
    except AttributeError:
        return -1


def sparse_coo_memory_usage(mat):
    try:
        return mat.data.nbytes + mat.row.nbytes + mat.col.nbytes
    except AttributeError:
        return -1


def sparse_other_memory_usage(a):
    try:
        return a.data.nbytes + a.indptr.nbytes + a.indices.nbytes
    except AttributeError:
        return -1


