import copy

try:
    file = open('eeee.txt', 'r')  # 会报错的代码
except Exception as e:  # 将报错存储在 e 中
    print(e)

pp = lambda x, y: x + y

x = 1
y = 2
print(pp(x, y))

a = [1, 2, 3]
print(id(a))
c = copy.copy(a)
print(id(c))
a[1] = 20
print(a)
print(c)



e=copy.deepcopy(a)
a[2]=333
print(e)
print(a)
