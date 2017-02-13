#!/bin/sh
#SBATCH --time=03:15:00          # Run time in hh:mm:ss
#SBATCH --mem-per-cpu=1024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=hello-world
#SBATCH --error=/work/scott/jamesd/job.%J.err
#SBATCH --output=/work/scott/jamesd/job.%J.out

module load example/test

hostname
sleep 60