import numpy as np

#可变参数
def report(name, *grades):
    total_grade = 0
    for grade in grades:
        total_grade += grade
    print(name, 'total grade is ', total_grade)

#关键字参数
def portrait(name, **kw):
    print('name is', name)
    for k,v in kw.items():
        print(k, v)

def opentext():
    myfile = open("./mytext.txt","a")
    myfile.write("I love java")
    myfile.close()


if __name__ == '__main__':
    #report("张三", 12, 13, 15, 17)
    portrait('Mike', age=24, country='China', education='bachelor')
    opentext()