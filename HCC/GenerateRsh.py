f = open('run.sh','w')
f.write('#!/bin/sh\n\n')
rndType = ['active','passive']
fold = [1,2,3,4,5,6,7,8,9,10]
add2Short = True
part1 = 'highmem'
part2 = False
partDict = { 1: part1, 2: part1, 3: part1, 4: part1, 5: part1,
6: part2, 7: part2, 8: part2, 9: part2, 10: part2 }
# "#SBATCH --partition=highmem\n" #tusker partition
# "#SBATCH --partition=gpu_k20\n" #crane partition
# "#SBATCH --partition=gpu_m2070\n"  #crane partition
# "#SBATCH --partition=guest\n"  #sandhills partition
cntShrt = 1
runDir = 'results11SclBy1_15'
for type in rndType:
    for fld in fold:
        f.write(
        "sbatch <<'EOF'\n"
        "#!/bin/sh\n"
        "#SBATCH --time=5:00:00          # Run time in hh:mm:ss\n"
        "#SBATCH --nodes=1       # number of nodes\n"
        "#SBATCH --ntasks=1       # number of cores\n"
        "#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)\n")
        if(partDict[fld]):
            f.write("#SBATCH --partition="+partDict[fld]+"\n")
        f.write(
        "#SBATCH --job-name=AP_" + type[0] + "_" + str(fld) + "\n"
        "#SBATCH --error=/work/scott/jamesd/"+runDir+"/log/job.%J.AP_" + type + "_" + str(fld) + ".err\n"
        "#SBATCH --output=/work/scott/jamesd/"+runDir+"/log/job.%J.AP_" + type + "_" + str(fld) + ".out\n")
        if(add2Short and cntShrt <=2):
            f.write("#SBATCH --qos=short\n")
            cntShrt+=1
        f.write(
        #"module load python/3.4\n" #for sandhills
        "module load python/3.5\n" #for crane and tusker
        "python /home/scott/jamesd/"+runDir+"/runActPass11.py " + type + " " + str(fld) + "\n"
        #"python /home/scott/jamesd/"+runDir+"/runActPass.py " + type + " " + str(fld) + "\n"
        #"python /home/scott/jamesd/"+runDir+"/runActPassRBF.py " + type + " " + str(fld) + "\n"
        "EOF\n\n")

f.close()

