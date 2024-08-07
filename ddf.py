import numpy as np

a = [1,3,5,7,9,11]
b = [3,5,6,11,12]

def cv(ls):
    return np.mean(ls)/np.std(ls)

print(cv(a))
print(cv(b))