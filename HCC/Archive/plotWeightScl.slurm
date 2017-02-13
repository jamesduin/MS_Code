#!/bin/sh
#SBATCH --time=00:05:00          # Run time in hh:mm:ss
#SBATCH --mem-per-cpu=1024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=opaguest
#SBATCH --job-name=plotWeightScl
#SBATCH --error=/work/scott/jamesd/job.%J.plotWeightScl.err
#SBATCH --output=/work/scott/jamesd/job.%J.plotWeightScl.out
#SBATCH --qos=short

module load python/3.5
python /home/scott/jamesd/plotWeightScl.py
