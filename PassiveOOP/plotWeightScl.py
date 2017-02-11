import numpy as np
from matplotlib import pyplot as plt
import pickle


plt.figure()
plt.style.use('ggplot')

x = np.arange(0.,30.,0.1)
def f(inp):
    y = np.array([25.0, 15.0])
    x = np.array([20.8870, 4.997])
    m = (y[0] - y[1]) / (x[0] - x[1])
    b = y[0] - m * x[0]
    return m*inp+b

plt.plot(x,f(x), label = 'scaleWeight')
print(x)
print(f(x))


plt.ylabel('PR-AUC')
plt.xlabel('Iteration')
plt.title('Active vs. Passive Learning')
plt.legend(loc="lower right")
plt.savefig('results/sampleWeight.png')

