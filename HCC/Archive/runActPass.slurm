#!/bin/sh
#SBATCH --time=01:00:00          # Run time in hh:mm:ss
#SBATCH --mem-per-cpu=60000       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=opaguest
#SBATCH --job-name=runActPass
#SBATCH --error=/work/scott/jamesd/job.%J.runActPass.err
#SBATCH --output=/work/scott/jamesd/job.%J.runActPass.out
#SBATCH --qos=short

module load python/3.5
python /home/scott/jamesd/MS_Code/ActivePassiveFolds/runActPass.py
