f = open('run.sh','w')
f.write('#!/bin/sh\n\n')
#f.write('\n\n')
rndType = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']
#rndType = ['0.0']
fold = [1,2,3,4,5,6,7,8,9,10]
add2Short = False
part1 = 'gpu_k20'
part2 = 'gpu_k20'
part3 = 'gpu_m2070'
partDict = { 1: part1, 2: part1, 3: part1, 4: part1, 5: part3,
6: part3, 7: part3, 8: part3, 9: part3, 10: part3 }
# "#SBATCH --partition=highmem\n" #tusker partition
# "#SBATCH --partition=gpu_k20\n" #crane partition
# "#SBATCH --partition=gpu_m2070\n"  #crane partition
# "#SBATCH --partition=guest\n"  #sandhills partition
cntShrt = 1

#cost = '1.1'
#cost = '1.2'
#cost = '1.5'
#cost = '32.0'
cost = '64.0'
#runDir = 'runFFRParam_Cst8'

runDir = 'runFFRParam'+ '_' + cost.replace('.', 'p')

for type in rndType:
    type1 = type.split(".")[0]
    type2 = type.split(".")[1]
    for fld in fold:
        f.write(
        "sbatch <<'EOF'\n"
        "#!/bin/sh\n"
        "#SBATCH --time=20:00:00          # Run time in hh:mm:ss\n"
        "#SBATCH --nodes=1       # number of nodes\n"
        "#SBATCH --ntasks=1       # number of cores\n"
        "#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)\n")
        if(partDict[fld]):
            f.write("#SBATCH --partition="+partDict[fld]+"\n")
        f.write(
        "#SBATCH --job-name=F" + type1+"p"+type2 + "_" + str(fld) + "\n"
        "#SBATCH --error=/work/scott/jamesd/"+runDir+"/log/job.%J.FFR_"  + type1+"p"+type2 + "_" + str(fld) + ".err\n"
        "#SBATCH --output=/work/scott/jamesd/"+runDir+"/log/job.%J.FFR_"  + type1+"p"+type2 + "_" + str(fld) + ".out\n")
        if(add2Short and cntShrt <=2):
            f.write("#SBATCH --qos=short\n")
            cntShrt+=1
        f.write(
        "module load python/3.5\n" #for crane and tusker
        "python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam "+cost+' '+type+' '+str(fld)+"\n"
        "EOF\n\n")

f.close()