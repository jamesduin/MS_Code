import numpy as np
from matplotlib import pyplot as plt
import pickle


plt.figure()
plt.style.use('ggplot')

scale = 4
print(scale)
#yorig = np.array([25., 15.])
yorig = np.array([20., 6.5])
print(yorig)
y = yorig*(scale)
print(y.tolist())
x = np.array([20.8870, 4.997])

inp = np.arange(0.,30.,0.1)
def f(inp):
    m = (y[0] - y[1]) / (x[0] - x[1])
    b = y[0] - m * x[0]
    return m*inp+b

plt.plot(inp,f(inp), label = 'scaleWeight')
plt.ylabel('PR-AUC')
plt.xlabel('Iteration')
plt.title('Active vs. Passive Learning')
plt.legend(loc="lower right")
plt.savefig('results/sampleWeight.png')

print(f(x))

print((np.array([1,0.5,0.9,0.75    ,4,0.9,2,1])*1/scale).tolist())
print(np.array([1,0.5,0.9,0.75    ,4,0.9,2,1])*1/scale*f(x[0]))
print(np.array([1,0.5,0.9,0.75    ,4,0.9,2,1])*1/scale*f(x[1]))
