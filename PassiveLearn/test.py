import os
import shutil

dir = 'ScalingDim/SVMMinMax'
if not os.path.exists(dir):
    os.makedirs(dir)
else:
    shutil.rmtree(dir)  # removes all the subdirectories!
    os.makedirs(dir)
os.makedirs(dir+'/coarse_results')
os.makedirs(dir+'/fine_results')
os.makedirs(dir+'/results')

os.chdir('ScalingDim/SVMMinMax')