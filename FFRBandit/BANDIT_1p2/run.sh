#!/bin/sh

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=B_1p2_1
#SBATCH --error=/work/scott/jamesd/BANDIT_1p2/log/job.%J.BT_1p2_1.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p2/log/job.%J.BT_1p2_1.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.2 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=B_1p2_2
#SBATCH --error=/work/scott/jamesd/BANDIT_1p2/log/job.%J.BT_1p2_2.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p2/log/job.%J.BT_1p2_2.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.2 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=B_1p2_3
#SBATCH --error=/work/scott/jamesd/BANDIT_1p2/log/job.%J.BT_1p2_3.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p2/log/job.%J.BT_1p2_3.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.2 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_k20
#SBATCH --job-name=B_1p2_4
#SBATCH --error=/work/scott/jamesd/BANDIT_1p2/log/job.%J.BT_1p2_4.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p2/log/job.%J.BT_1p2_4.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.2 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_k20
#SBATCH --job-name=B_1p2_5
#SBATCH --error=/work/scott/jamesd/BANDIT_1p2/log/job.%J.BT_1p2_5.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p2/log/job.%J.BT_1p2_5.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.2 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_k20
#SBATCH --job-name=B_1p2_6
#SBATCH --error=/work/scott/jamesd/BANDIT_1p2/log/job.%J.BT_1p2_6.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p2/log/job.%J.BT_1p2_6.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.2 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_k20
#SBATCH --job-name=B_1p2_7
#SBATCH --error=/work/scott/jamesd/BANDIT_1p2/log/job.%J.BT_1p2_7.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p2/log/job.%J.BT_1p2_7.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.2 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_k20
#SBATCH --job-name=B_1p2_8
#SBATCH --error=/work/scott/jamesd/BANDIT_1p2/log/job.%J.BT_1p2_8.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p2/log/job.%J.BT_1p2_8.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.2 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_k20
#SBATCH --job-name=B_1p2_9
#SBATCH --error=/work/scott/jamesd/BANDIT_1p2/log/job.%J.BT_1p2_9.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p2/log/job.%J.BT_1p2_9.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.2 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_k20
#SBATCH --job-name=B_1p2_10
#SBATCH --error=/work/scott/jamesd/BANDIT_1p2/log/job.%J.BT_1p2_10.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p2/log/job.%J.BT_1p2_10.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.2 10
EOF

