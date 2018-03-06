#cython: infer_types=True, wraparound=False, boundscheck=False

def mse(object arr1 not None, object arr2 not None):

    arr1 = arr1.reshape(-1)
    arr2 = arr2.reshape(-1)
    cdef int[::1] arr1_mem = arr1
    cdef int[::1] arr2_mem = arr2
    cdef int tmp
    total = 0
    for i in range(arr1_mem.shape[0]):
        tmp = arr1_mem[i] - arr2_mem[i]
        total += tmp**2

    result = <double>total / arr1_mem.size
    return result

