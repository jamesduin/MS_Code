import re
import os
import time


st = time.perf_counter()

time.sleep(1.22)

end = time.perf_counter()
tot = 3833333.37
# print(divmod(tot,60)[1])
# print(divmod(tot,60))
#tot = end-st
print('{:.0f}hr {:.0f}m {:.2f}sec'.format(*divmod(divmod(tot,60)[0],60),divmod(tot,60)[1]))
