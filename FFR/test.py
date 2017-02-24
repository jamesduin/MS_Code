import re
import os
import time
import sys
import random
import itertools
import bisect
import pickle


results = pickle.load(open('runFFRR_Cst16/results/FFR_1p0_2.res', 'rb'))

f = open('runFFRR_Cst16/results/_out.txt', 'w')
for rec in results:
    f.write(str(rec) + '\n')
f.close()