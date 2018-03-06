import time

def time_func(func, name, *args, **kwargs):
    cdef int nb_runs
    t = time.time()
    for _ in range(10):
        func(*args, **kwargs)
    t2 = time.time()

    # We would like the test to last 1s max.

    estimated = (t2-t)/10
    nb_runs = 2/estimated

    t = time.time()
    for _ in range(nb_runs):
        func(*args, **kwargs)
    t2 = time.time()

    print(nb_runs, 'runs for ', name)
    print('execution time: ', 1000*(t2-t)/nb_runs, 'ms')
    print('result = ', func(*args, **kwargs))
