import time


class Calculator:
    name='good calculator'
    price=18
    def __init__(self,name,price,hight=10,width=14,weight=16): #后面三个属性设置默认值,查看运行
        self.name=name
        self.price=price
        self.h=hight
        self.wi=width
        self.we=weight

def fib(max):
    a, b = 0, 1
    while max:
        r = b
        a, b = b, a+b
        max -= 1
        yield r

if __name__ == '__main__':
    c = Calculator('bad calculator', 18)
    print(c.h)
    print(c.we)
    print(c.wi)

    dic = {}
    dic['lan'] = 'python'
    dic['version'] = 2.7
    dic['platform'] = 64
    for key in dic:
        print(key, dic[key])

    for i in fib(5):
        print(i)

    print(time.localtime())