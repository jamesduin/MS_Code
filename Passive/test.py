import re
file = re.split("[/\.]",__file__)[-2]
print(re.split("_",file)[1])
