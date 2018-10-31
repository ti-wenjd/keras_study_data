text = 'This is my first test.\nThis is the second line.\nThis the third line'

##写文件
my_file = open("./my_file.txt", "w")
my_file.write(text)
my_file.close()

##读文件  使用 file.read() 能够读取到文本的所有内容.
read_file = open("./my_file.txt", "r")
content = read_file.read()
print(content)

# 按行读取 file.readline()   ,
# 可以使用 file.readline(), file.readline()
# 读取的内容和你使用的次数有关, 使用第二次的时候, 读取到的是文本的第二行, 并可以以此类推:
read_file1 = open("./my_file.txt", "r")
first_row = read_file1.readline()
print(first_row)

#如果想要读取所有行, 并可以使用像 for 一样的迭代器迭代这些行结果,
# 我们可以使用 file.readlines(), 将每一行的结果存储在 list 中, 方便以后迭代.
read_file2 = open("./my_file.txt", "r")
all_rows = read_file2.readlines()
print(all_rows)