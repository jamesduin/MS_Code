f = open('run.sh','w')
f.write('#!/bin/sh\n\n')
rndType = ['active','passive']
fold = [7,8,9,10]
for type in rndType:
    for fld in fold:
        f.write(
        "sbatch <<'EOF'\n"
        "#!/bin/sh\n"
        "#SBATCH --time=05:00:00          # Run time in hh:mm:ss\n"
        "#SBATCH --mem-per-cpu=60000       # Maximum memory required per CPU (in megabytes)\n"
        "#SBATCH --partition=opaguest\n"
        "#SBATCH --job-name=runActPass\n"
        "#SBATCH --error=/work/scott/jamesd/job.%J.runActPass_" + type + "_" + str(fld) + ".err\n"
        "#SBATCH --output=/work/scott/jamesd/job.%J.runActPass_" + type + "_" + str(fld) + ".out\n"
        "#SBATCH --qos=short\n"
        "module load python/3.5\n"
        "python /home/scott/jamesd/MS_Code/ActivePassiveFolds/runActPass.py " + type + " " + str(fld) + "\n"
        "EOF\n\n"
        )

        print(
        "sbatch <<'EOF'\n"
        "#!/bin/sh\n"
        "#SBATCH --time=05:00:00          # Run time in hh:mm:ss\n"
        "#SBATCH --mem-per-cpu=60000       # Maximum memory required per CPU (in megabytes)\n"
        "#SBATCH --partition=opaguest\n"
        "#SBATCH --job-name=runActPass\n"
        "#SBATCH --error=/work/scott/jamesd/job.%J.runActPass_"+type+"_"+str(fld)+".err\n"
        "#SBATCH --output=/work/scott/jamesd/job.%J.runActPass_"+type+"_"+str(fld)+".out\n"
        "#SBATCH --qos=short\n"
        "module load python/3.5\n"
        "python /home/scott/jamesd/MS_Code/ActivePassiveFolds/runActPass.py "+type+" "+str(fld)+"\n"
        "EOF\n\n")

f.close()

