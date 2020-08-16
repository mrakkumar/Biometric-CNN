from __future__ import print_function

import numpy as np
import CNN_test_hardware as test


with open('picdata.txt', 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8)
# The inputs are vectors now, we reshape them to monochrome 2D images,
# following the shape convention: (examples, channels, rows, columns)
data = data.reshape(-1, 1, 45,60)
# The inputs come as bytes, we convert them to float32 in range [0,1].
data = data / np.float32(256)
np.set_printoptions(threshold=np.inf)
acc=test.identify(data)
print (acc)