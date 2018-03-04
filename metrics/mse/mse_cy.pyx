import cython

@cython.wraparound(False)
@cython.boundscheck(False)
def mse(object arr1 not None, object arr2 not None):

    arr1 = arr1.reshape(-1)
    arr2 = arr2.reshape(-1)
    cdef int[::1] arr1_mem = arr1
    cdef int[::1] arr2_mem = arr2

    cdef int diff, total = 0
    for i in range(arr1_mem.shape[0]):
        diff = (arr1_mem[i] - arr2_mem[i])
        total += diff**2


    cdef double result = total / arr1_mem.size
    return result

