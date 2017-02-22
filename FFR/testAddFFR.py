import re
import os
import time
import sys
import random
import itertools
import bisect
from decimal import *
import numpy as np
getcontext().prec = 8


batch = Decimal(100.0)
fineCost = Decimal(16.0)
coarseCost = Decimal(1.0)
FFRList = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#FFRList = [0,1,2,3,4,5,6,7,8,9,1]

def retAddNum(add):
    remain = add % Decimal(1.0)
    num = int(add-remain)
    if(remain >0):
        x = Decimal(random.random())
        if(x < remain):
            num+= 1
    return num

for FFR in FFRList:
    add = dict()
    add['fine'] = float(batch*(Decimal(FFR))/fineCost)
    add['coarse'] = float(batch*(Decimal(1.0)-Decimal(FFR))/coarseCost)
    print(add)
    for i in range(10):
        pick = dict()
        pick['fine'] = retAddNum(batch*(Decimal(FFR))/fineCost)
        pick['coarse'] = retAddNum(batch*(Decimal(1.0)-Decimal(FFR))/coarseCost)
        print(pick)
    stat = dict()
    nums = []
    for i in range(1000):
        nums.append(retAddNum(batch*(Decimal(FFR))/fineCost))
        stat['fine'] = np.sum(nums) / len(nums)
    nums = []
    for i in range(1000):
        nums.append(retAddNum(batch*(Decimal(1.0)-Decimal(FFR))/coarseCost))
        stat['coarse'] = np.sum(nums) / len(nums)
    print(stat)
    print('\n')




#
# ratio = 0.625
# print(ratio)
# nums = []
# for i in range(100000):
#     x = random.random()
#     #cumdist = [ratio]
#     #choices = [0,1]
#     #print(choices[bisect.bisect(cumdist, x)])
#     #nums.append(choices[bisect.bisect(cumdist, x)])
#     if(x >= ratio):
#         nums.append(0.0)
#     else:
#         nums.append(1.0)
#
# print(np.sum(nums)/len(nums))
# print(np.sum(nums))
# print(len(nums))


# weighted_choices = [('Red', 3), ('Blue', 2), ('Yellow', 1), ('Green', 4)]
# print(weighted_choices)
# print(*weighted_choices)
# choices, weights = zip(*weighted_choices)
# cumdist = list(itertools.accumulate(weights))
# print(cumdist)            # [3, 3+2, 3+2+1, 3+2+1+4]
# x = random.random() * cumdist[-1]
# print(choices)
# print(choices[bisect.bisect(cumdist, x)])