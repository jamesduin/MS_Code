f = open('run.sh','w')
f.write('#!/bin/sh\n\n')
#rndType = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']
fold = [1,2,3,4,5,6,7,8,9,10]
add2Short = True
part1 = 'gpu_m2070'
part2 = 'gpu_k20'
part3 = False
partDict = { 1: part1, 2: part1, 3: part2, 4: part2, 5: part3,
6: part3, 7: part3, 8: part3, 9: part3, 10: part3 }
# "#SBATCH --partition=highmem\n" #tusker partition
# "#SBATCH --partition=gpu_k20\n" #crane partition
# "#SBATCH --partition=gpu_m2070\n"  #crane partition
# "#SBATCH --partition=guest\n"  #sandhills partition
cntShrt = 1
runDir = 'Bandit_RandEqual_Cst1'
#for type in rndType:
for fld in fold:
    f.write(
    "sbatch <<'EOF'\n"
    "#!/bin/sh\n"
    "#SBATCH --time=3:00:00          # Run time in hh:mm:ss\n"
    "#SBATCH --nodes=1       # number of nodes\n"
    "#SBATCH --ntasks=1       # number of cores\n"
    "#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)\n")
    if(partDict[fld]):
        f.write("#SBATCH --partition="+partDict[fld]+"\n")
    f.write(
    "#SBATCH --job-name=BT_" + str(fld) + "\n"
    "#SBATCH --error=/work/scott/jamesd/"+runDir+"/log/job.%J.BT_" + str(fld) + ".err\n"
    "#SBATCH --output=/work/scott/jamesd/"+runDir+"/log/job.%J.BT_" + str(fld) + ".out\n")
    if(add2Short and cntShrt <=2):
        f.write("#SBATCH --qos=short\n")
        cntShrt+=1
    f.write(
    "module load python/3.5\n" #for crane and tusker
    "python /home/scott/jamesd/"+runDir+"/runFFRBandit.py " + str(fld) + "\n"
    "EOF\n\n")

f.close()