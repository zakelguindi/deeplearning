import numpy as np 
a = np.random.randn(5)

print(a)
print(a.shape)
print(a.T)
print(np.dot(a,a.T))

print('-------------------\n')
a = np.random.randn(5,1)
print(a.T)
print(np.dot(a, a.T))