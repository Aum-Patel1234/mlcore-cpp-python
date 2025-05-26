import sys
import os
import numpy as np

sys.path.append(os.path.abspath("../cpp/build"))

import mlcore_cpp  # Your module name here

print(dir(mlcore_cpp))

arr = np.array([5, 1, 2, 3], dtype=np.int32)
wrapper = mlcore_cpp.NdarrayWrapperUInt8(arr)
print(wrapper.size())
print(wrapper.dtype())
