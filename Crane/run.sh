#!/bin/sh

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=05:00:00          # Run time in hh:mm:ss
#SBATCH --mem-per-cpu=60000       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=opaguest
#SBATCH --job-name=runActPass
#SBATCH --error=/work/scott/jamesd/job.%J.runActPass_active_9.err
#SBATCH --output=/work/scott/jamesd/job.%J.runActPass_active_9.out
#SBATCH --qos=short
module load python/3.5
python /home/scott/jamesd/MS_Code/ActivePassiveFolds/runActPass.py active 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=05:00:00          # Run time in hh:mm:ss
#SBATCH --mem-per-cpu=60000       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=opaguest
#SBATCH --job-name=runActPass
#SBATCH --error=/work/scott/jamesd/job.%J.runActPass_active_10.err
#SBATCH --output=/work/scott/jamesd/job.%J.runActPass_active_10.out
#SBATCH --qos=short
module load python/3.5
python /home/scott/jamesd/MS_Code/ActivePassiveFolds/runActPass.py active 10
EOF


sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=05:00:00          # Run time in hh:mm:ss
#SBATCH --mem-per-cpu=60000       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=opaguest
#SBATCH --job-name=runActPass
#SBATCH --error=/work/scott/jamesd/job.%J.runActPass_passive_8.err
#SBATCH --output=/work/scott/jamesd/job.%J.runActPass_passive_8.out
#SBATCH --qos=short
module load python/3.5
python /home/scott/jamesd/MS_Code/ActivePassiveFolds/runActPass.py passive 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=05:00:00          # Run time in hh:mm:ss
#SBATCH --mem-per-cpu=60000       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=opaguest
#SBATCH --job-name=runActPass
#SBATCH --error=/work/scott/jamesd/job.%J.runActPass_passive_9.err
#SBATCH --output=/work/scott/jamesd/job.%J.runActPass_passive_9.out
#SBATCH --qos=short
module load python/3.5
python /home/scott/jamesd/MS_Code/ActivePassiveFolds/runActPass.py passive 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=05:00:00          # Run time in hh:mm:ss
#SBATCH --mem-per-cpu=60000       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=opaguest
#SBATCH --job-name=runActPass
#SBATCH --error=/work/scott/jamesd/job.%J.runActPass_passive_10.err
#SBATCH --output=/work/scott/jamesd/job.%J.runActPass_passive_10.out
#SBATCH --qos=short
module load python/3.5
python /home/scott/jamesd/MS_Code/ActivePassiveFolds/runActPass.py passive 10
EOF

