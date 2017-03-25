#!/bin/sh

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=B_1p5_1
#SBATCH --error=/work/scott/jamesd/BANDIT_1p5/log/job.%J.BT_1p5_1.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p5/log/job.%J.BT_1p5_1.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.5 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=B_1p5_2
#SBATCH --error=/work/scott/jamesd/BANDIT_1p5/log/job.%J.BT_1p5_2.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p5/log/job.%J.BT_1p5_2.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.5 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=B_1p5_3
#SBATCH --error=/work/scott/jamesd/BANDIT_1p5/log/job.%J.BT_1p5_3.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p5/log/job.%J.BT_1p5_3.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.5 3
EOF


sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=B_1p5_4
#SBATCH --error=/work/scott/jamesd/BANDIT_1p5/log/job.%J.BT_1p5_4.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p5/log/job.%J.BT_1p5_4.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.5 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=B_1p5_5
#SBATCH --error=/work/scott/jamesd/BANDIT_1p5/log/job.%J.BT_1p5_5.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p5/log/job.%J.BT_1p5_5.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.5 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=B_1p5_6
#SBATCH --error=/work/scott/jamesd/BANDIT_1p5/log/job.%J.BT_1p5_6.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p5/log/job.%J.BT_1p5_6.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.5 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=B_1p5_7
#SBATCH --error=/work/scott/jamesd/BANDIT_1p5/log/job.%J.BT_1p5_7.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p5/log/job.%J.BT_1p5_7.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.5 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=B_1p5_8
#SBATCH --error=/work/scott/jamesd/BANDIT_1p5/log/job.%J.BT_1p5_8.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p5/log/job.%J.BT_1p5_8.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.5 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=B_1p5_9
#SBATCH --error=/work/scott/jamesd/BANDIT_1p5/log/job.%J.BT_1p5_9.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p5/log/job.%J.BT_1p5_9.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.5 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=B_1p5_10
#SBATCH --error=/work/scott/jamesd/BANDIT_1p5/log/job.%J.BT_1p5_10.err
#SBATCH --output=/work/scott/jamesd/BANDIT_1p5/log/job.%J.BT_1p5_10.out
module load python/3.5
python /home/scott/jamesd/BANDIT/runFFRBandit.py BANDIT 1.5 10
EOF

