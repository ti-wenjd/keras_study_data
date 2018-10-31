import multiprocessing as mp


def job(x):
    return x * x


# 1.Pool默认调用是CPU的核数，传入processes参数可自定义CPU核数
# 2.map() 放入迭代参数，返回多个结果
# 3.apply_async()只能放入一组参数，并返回一个结果，如果想得到map()的效果需要通过迭代

def multicore():
    pool = mp.Pool()
    res = pool.map(job, range(10))
    print(res)
    res = pool.apply_async(job, (2,))
    # 用get获得结果
    print(res.get())
    # 迭代器，i=0时apply一次，i=1时apply一次等等
    multi_res = [pool.apply_async(job, (i,)) for i in range(10)]
    # 从迭代器中取出
    print([res.get() for res in multi_res])


if __name__ == '__main__':
    multicore()
