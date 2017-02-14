#!/bin/sh

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=02:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=AP_a_6
#SBATCH --error=/work/scott/jamesd/resultsSclBy1_4375/job.%J.AP_active_6.err
#SBATCH --output=/work/scott/jamesd/resultsSclBy1_4375/job.%J.AP_active_6.out
module load python/3.4
python /home/scott/jamesd/resultsSclBy1_4375/runActPass.py active 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=02:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=AP_a_7
#SBATCH --error=/work/scott/jamesd/resultsSclBy1_4375/job.%J.AP_active_7.err
#SBATCH --output=/work/scott/jamesd/resultsSclBy1_4375/job.%J.AP_active_7.out
module load python/3.4
python /home/scott/jamesd/resultsSclBy1_4375/runActPass.py active 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=02:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=AP_a_8
#SBATCH --error=/work/scott/jamesd/resultsSclBy1_4375/job.%J.AP_active_8.err
#SBATCH --output=/work/scott/jamesd/resultsSclBy1_4375/job.%J.AP_active_8.out
module load python/3.4
python /home/scott/jamesd/resultsSclBy1_4375/runActPass.py active 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=02:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=AP_a_9
#SBATCH --error=/work/scott/jamesd/resultsSclBy1_4375/job.%J.AP_active_9.err
#SBATCH --output=/work/scott/jamesd/resultsSclBy1_4375/job.%J.AP_active_9.out
module load python/3.4
python /home/scott/jamesd/resultsSclBy1_4375/runActPass.py active 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=02:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=AP_a_10
#SBATCH --error=/work/scott/jamesd/resultsSclBy1_4375/job.%J.AP_active_10.err
#SBATCH --output=/work/scott/jamesd/resultsSclBy1_4375/job.%J.AP_active_10.out
module load python/3.4
python /home/scott/jamesd/resultsSclBy1_4375/runActPass.py active 10
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=02:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=AP_p_6
#SBATCH --error=/work/scott/jamesd/resultsSclBy1_4375/job.%J.AP_passive_6.err
#SBATCH --output=/work/scott/jamesd/resultsSclBy1_4375/job.%J.AP_passive_6.out
module load python/3.4
python /home/scott/jamesd/resultsSclBy1_4375/runActPass.py passive 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=02:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=AP_p_7
#SBATCH --partition=guest
#SBATCH --error=/work/scott/jamesd/resultsSclBy1_4375/job.%J.AP_passive_7.err
#SBATCH --output=/work/scott/jamesd/resultsSclBy1_4375/job.%J.AP_passive_7.out
module load python/3.4
python /home/scott/jamesd/resultsSclBy1_4375/runActPass.py passive 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=02:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=AP_p_8
#SBATCH --partition=guest
#SBATCH --error=/work/scott/jamesd/resultsSclBy1_4375/job.%J.AP_passive_8.err
#SBATCH --output=/work/scott/jamesd/resultsSclBy1_4375/job.%J.AP_passive_8.out
module load python/3.4
python /home/scott/jamesd/resultsSclBy1_4375/runActPass.py passive 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=02:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=AP_p_9
#SBATCH --partition=guest
#SBATCH --error=/work/scott/jamesd/resultsSclBy1_4375/job.%J.AP_passive_9.err
#SBATCH --output=/work/scott/jamesd/resultsSclBy1_4375/job.%J.AP_passive_9.out
module load python/3.4
python /home/scott/jamesd/resultsSclBy1_4375/runActPass.py passive 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=02:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=AP_p_10
#SBATCH --partition=guest
#SBATCH --error=/work/scott/jamesd/resultsSclBy1_4375/job.%J.AP_passive_10.err
#SBATCH --output=/work/scott/jamesd/resultsSclBy1_4375/job.%J.AP_passive_10.out
module load python/3.4
python /home/scott/jamesd/resultsSclBy1_4375/runActPass.py passive 10
EOF

