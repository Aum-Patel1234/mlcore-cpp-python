import sys
import os
import numpy as np

sys.path.append(os.path.abspath("../cpp/build"))

import mlcore_cpp  # Your module name here

print(dir(mlcore_cpp))

arr = np.array(
    [
        [5, 1],
        [2, 4],
    ],
    dtype=np.int32,
)
wrapper = mlcore_cpp.NdarrayWrapperInt8(arr)
print(wrapper.dtype())
print(wrapper.size())
print(wrapper.shape())
print(wrapper.ndim())
print(wrapper.get())
