import numpy

dic = {'1': [2], '2': [[2, 3, 4, 5], [1, 2]]}

a = list(dic.items())
print(a[1][1])
b = [[1], [2, 23]]
b = numpy.array(b)
print(sum(b, []))
print(float(inf))