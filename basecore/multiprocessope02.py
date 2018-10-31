import multiprocessing as mp


def job(x):
    return x * x


#进程池 Pool() 和 map()
def multicore():
    pool = mp.Pool()
    res = pool.map(job, range(10))
    print(res)


if __name__ == '__main__':
    multicore()
