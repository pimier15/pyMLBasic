import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.0001 , 1.5 , 0.0001)

f = x*np.log2(x)

plt.plot(x,f)
plt.show()


