import sys
import os
import numpy as np

sys.path.append(os.path.abspath("../../cpp/build"))

import mlcore_cpp

arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.int32)
print(arr.dtype)
mlcore_cpp.print_array_eg(arr)
