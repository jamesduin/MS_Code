#!/bin/sh

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=B_1p1_1
#SBATCH --error=/work/scott/jamesd/BANDIT_1p1/log/job.%J.BT_1p1_1.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p1/log/job.%J.BT_1p1_1.out
#SBATCH --qos=short
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.1 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=B_1p1_2
#SBATCH --error=/work/scott/jamesd/BANDIT_1p1/log/job.%J.BT_1p1_2.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p1/log/job.%J.BT_1p1_2.out
#SBATCH --qos=short
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.1 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=B_1p1_3
#SBATCH --error=/work/scott/jamesd/BANDIT_1p1/log/job.%J.BT_1p1_3.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p1/log/job.%J.BT_1p1_3.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.1 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_k20
#SBATCH --job-name=B_1p1_4
#SBATCH --error=/work/scott/jamesd/BANDIT_1p1/log/job.%J.BT_1p1_4.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p1/log/job.%J.BT_1p1_4.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.1 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_k20
#SBATCH --job-name=B_1p1_5
#SBATCH --error=/work/scott/jamesd/BANDIT_1p1/log/job.%J.BT_1p1_5.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p1/log/job.%J.BT_1p1_5.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.1 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_k20
#SBATCH --job-name=B_1p1_6
#SBATCH --error=/work/scott/jamesd/BANDIT_1p1/log/job.%J.BT_1p1_6.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p1/log/job.%J.BT_1p1_6.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.1 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=B_1p1_7
#SBATCH --error=/work/scott/jamesd/BANDIT_1p1/log/job.%J.BT_1p1_7.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p1/log/job.%J.BT_1p1_7.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.1 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=B_1p1_8
#SBATCH --error=/work/scott/jamesd/BANDIT_1p1/log/job.%J.BT_1p1_8.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p1/log/job.%J.BT_1p1_8.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.1 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=B_1p1_9
#SBATCH --error=/work/scott/jamesd/BANDIT_1p1/log/job.%J.BT_1p1_9.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p1/log/job.%J.BT_1p1_9.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.1 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=B_1p1_10
#SBATCH --error=/work/scott/jamesd/BANDIT_1p1/log/job.%J.BT_1p1_10.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p1/log/job.%J.BT_1p1_10.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.1 10
EOF

