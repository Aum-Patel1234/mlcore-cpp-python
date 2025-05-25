import sys
import os, gc

sys.path.append(os.path.abspath("../../cpp/build"))

import mlcore_cpp

obj = mlcore_cpp.Example()

del obj
gc.collect()

print("Done")
sys.stdout.flush()
