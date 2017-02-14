import re
import os
rootDir = re.split('[/\.]',__file__)[1]
if(rootDir == 'Users'):
    dataDir = '../'
else:
    os.chdir('/work/scott/jamesd/')
    dataDir = '/home/scott/jamesd/MS_Code/'
