#!/bin/sh
#SBATCH --time=00:05:00          # Run time in hh:mm:ss
#SBATCH --mem-per-cpu=1024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=test-python
#SBATCH --error=/work/scott/jamesd/job.%J.testpy.err
#SBATCH --output=/work/scott/jamesd/job.%J.testpy.out

module load python/3.5
python /home/scott/jamesd/test.py
