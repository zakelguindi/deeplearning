import numpy as np 
import time 
a = np.array([1, 2, 3, 4])
a = np.random.rand(1000000)
b = np.random.rand(1000000)

# --------------vectorized version is 300x faster than for-loop -------------------------------

tic = time.time()
c = np.dot(a, b)
toc = time.time()

print(f"vectorized version: {1000*(toc-tic)}ms")
print(c)
c = 0
tic = time.time()
for i in range(1000000):
    c += a[i] * b[i]
toc = time.time()
print(f"For loop: {1000*(toc-tic)}ms")
print(c)