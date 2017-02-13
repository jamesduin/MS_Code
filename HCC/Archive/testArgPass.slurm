#!/bin/sh
#SBATCH --time=05:00:00          # Run time in hh:mm:ss
#SBATCH --mem-per-cpu=60000       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=opaguest
#SBATCH --job-name=runActPass
#SBATCH --error=/work/scott/jamesd/job.%J.testArgPass$type_$fold.err
#SBATCH --output=/work/scott/jamesd/job.%J.runActPass$type_$fold.out
#SBATCH --qos=short

module load python/3.5
python /home/scott/jamesd/test.py $type $fold
