import sys
import os
import numpy as np
import time

sys.path.append(os.path.abspath("../cpp/build"))

import mlcore_cpp  # Your module name here

print(dir(mlcore_cpp))

arr = np.array(
    [
        [5, 1, 4],
        [2, 4, 3],
    ],
)
wrapper = mlcore_cpp.NdarrayWrapperInt8(arr)
print(wrapper.dtype())
print(wrapper.size())
print(wrapper.shape())
print(wrapper.ndim())
print("Array made by xtensor - ", wrapper.get(), "\n")
print("normal vec - ", wrapper.getVec())
wrapper.cpp_forloop()

start = time.time()

for i in range(10000000):
    pass

end = time.time()
print(f"Python Time taken: {(end - start) * 1000:.2f} ms")

# ['NdarrayWrapperDouble', 'NdarrayWrapperFloat', 'NdarrayWrapperInt16', 'NdarrayWrapperInt32', 'NdarrayWrapperInt64', 'NdarrayWrapperInt8', 'NdarrayWrapperLongDouble', 'NdarrayWrapperUI
# nt16', 'NdarrayWrapperUInt32', 'NdarrayWrapperUInt64', 'NdarrayWrapperUInt8', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__']
# int8_t
# 4
# [2, 2]
# 2
# [5, 1, 2, 4]
# C++ Time taken: 29 ms
# Python Time taken: 514.56 ms
