import numpy as np
import matplotlib.pyplot as plt

p = np.arange(0.01,0.9,0.01)

fp = p / (1.0-p)

logfp = np.log(fp)

logfpT = np.reciprocal(logfp)

#plt.subplots(2,2)
#plt.hold(True)
#plt.xlabel("p")
#
#plt.subplot(221)
#plt.plot(p,fp)
#
#plt.subplot(222)
#plt.plot(p,logfp)
#plt.show()
#
#
#plt.subplot(223)
plt.plot(p,logfpT)
plt.show()

