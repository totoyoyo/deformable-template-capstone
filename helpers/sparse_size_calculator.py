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


### Junk
# from helpers.sparse_size_calculator import sparse_dia_memory_usage, \
#     sparse_coo_memory_usage, sparse_other_memory_usage
#
# import scipy.sparse.linalg as ssl
# normal_size = SIGMA_P_INV.nbytes
# COO_P = ss.coo_matrix(SIGMA_P_INV)
# CSC_P = ss.csc_matrix(SIGMA_P_INV)
# CSR_P = ss.csr_matrix(SIGMA_P_INV)
# DIA_P = ss.dia_matrix(SIGMA_P_INV)
# dia_s = sparse_dia_memory_usage(DIA_P)
# coo_s = sparse_coo_memory_usage(COO_P)
# csr_s = sparse_other_memory_usage(CSR_P)
# csc_s = sparse_other_memory_usage(CSC_P)
#
# I_COO_P = ssl.inv(COO_P)
# I_CSR_P = ssl.inv(CSR_P)
# I_CSC_P = ssl.inv(CSC_P)
# I_DIA_P = ssl.inv(DIA_P)
# idia_s = sparse_other_memory_usage(I_DIA_P)
# icoo_s = sparse_coo_memory_usage(I_COO_P)
# icsr_s = sparse_other_memory_usage(I_CSR_P)
# icsc_s = sparse_other_memory_usage(I_CSC_P)
