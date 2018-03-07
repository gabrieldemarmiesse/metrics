#cython: infer_types=True, wraparound=False, boundscheck=False, cdivision=True
from posix.time cimport CLOCK_REALTIME, timespec, clock_gettime

def time_func(func, name, *args, **kwargs):
    cdef timespec t1, t2
    clock_gettime(CLOCK_REALTIME, &t1)
    func(*args, **kwargs)
    clock_gettime(CLOCK_REALTIME, &t2)

    # We would like the test to last 2s max.
    estimated = ms_difference(&t1, &t2)
    estimated_s = estimated/1000
    nb_runs = max(1, <int>(2./estimated_s))

    clock_gettime(CLOCK_REALTIME, &t1)
    for _ in range(nb_runs):
        func(*args, **kwargs)
    clock_gettime(CLOCK_REALTIME, &t2)

    print(nb_runs, 'runs for ', name)
    print('execution time: ', ms_difference(&t1, &t2)/nb_runs, 'ms')
    print('result = ', func(*args, **kwargs))


cdef double ms_difference(timespec *t1, timespec *t2):
    nb_seconds = t2[0].tv_sec - t1[0].tv_sec
    cdef double nb_nseconds = t2[0].tv_nsec - t1[0].tv_nsec
    return nb_seconds*1000 + nb_nseconds/1e6
