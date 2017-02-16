f = open('run.sh','w')
f.write('#!/bin/sh\n\n')
fold = [0.0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
add2Short = False
part1 = 'highmem'
part2 = False
partDict = { 0.0: part1,0.1: part1, 0.2: part1, 0.3: part1, 0.4: part1, 0.5: part1,
0.6: part2, 0.7: part2, 0.8: part2, 0.9: part2, 1.0: part2 }
# "#SBATCH --partition=highmem\n" #tusker partition
# "#SBATCH --partition=gpu_k20\n" #crane partition
# "#SBATCH --partition=gpu_m2070\n"  #crane partition
# "#SBATCH --partition=guest\n"  #sandhills partition
cntShrt = 1
runDir = 'resultsFFR_1'

for fld in fold:
    f.write(
    "sbatch <<'EOF'\n"
    "#!/bin/sh\n"
    "#SBATCH --time=15:00:00          # Run time in hh:mm:ss\n"
    "#SBATCH --nodes=1       # number of nodes\n"
    "#SBATCH --ntasks=1       # number of cores\n"
    "#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)\n")
    if(partDict[fld]):
        f.write("#SBATCH --partition="+partDict[fld]+"\n")
    f.write(
    "#SBATCH --job-name=FFR_"+ str(fld).replace('.','_') + "\n"
    "#SBATCH --error=/work/scott/jamesd/"+runDir+"/log/job.%J.FFR_" + str(fld).replace('.','_') + ".err\n"
    "#SBATCH --output=/work/scott/jamesd/"+runDir+"/log/job.%J.FFR_" + str(fld).replace('.','_') + ".out\n")
    if(add2Short and cntShrt <=2):
        f.write("#SBATCH --qos=short\n")
        cntShrt+=1
    f.write(
    #"module load python/3.4\n" #for sandhills
    "module load python/3.5\n" #for crane and tusker
    #"python /home/scott/jamesd/"+runDir+"/runActPass11.py " + type + " " + str(fld) + "\n"
    #"python /home/scott/jamesd/"+runDir+"/runActPass.py " + type + " " + str(fld) + "\n"
    #"python /home/scott/jamesd/"+runDir+"/runActPassRBF.py " + type + " " + str(fld) + "\n"
    "python /home/scott/jamesd/"+runDir+"/runFFR.py " +str(fld) + "\n"
    "EOF\n\n")

f.close()

