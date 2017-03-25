#!/bin/sh

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=BT_1
#SBATCH --error=/work/scott/jamesd/Bandit_RandEqual_Cst1/log/job.%J.BT_1.err
#SBATCH --output=/work/scott/jamesd/Bandit_RandEqual_Cst1/log/job.%J.BT_1.out
#SBATCH --qos=short
module load python/3.5
python /home/scott/jamesd/Bandit_RandEqual_Cst1/runFFRBandit.py 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=BT_2
#SBATCH --error=/work/scott/jamesd/Bandit_RandEqual_Cst1/log/job.%J.BT_2.err
#SBATCH --output=/work/scott/jamesd/Bandit_RandEqual_Cst1/log/job.%J.BT_2.out
#SBATCH --qos=short
module load python/3.5
python /home/scott/jamesd/Bandit_RandEqual_Cst1/runFFRBandit.py 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_k20
#SBATCH --job-name=BT_3
#SBATCH --error=/work/scott/jamesd/Bandit_RandEqual_Cst1/log/job.%J.BT_3.err
#SBATCH --output=/work/scott/jamesd/Bandit_RandEqual_Cst1/log/job.%J.BT_3.out
module load python/3.5
python /home/scott/jamesd/Bandit_RandEqual_Cst1/runFFRBandit.py 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_k20
#SBATCH --job-name=BT_4
#SBATCH --error=/work/scott/jamesd/Bandit_RandEqual_Cst1/log/job.%J.BT_4.err
#SBATCH --output=/work/scott/jamesd/Bandit_RandEqual_Cst1/log/job.%J.BT_4.out
module load python/3.5
python /home/scott/jamesd/Bandit_RandEqual_Cst1/runFFRBandit.py 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=BT_5
#SBATCH --error=/work/scott/jamesd/Bandit_RandEqual_Cst1/log/job.%J.BT_5.err
#SBATCH --output=/work/scott/jamesd/Bandit_RandEqual_Cst1/log/job.%J.BT_5.out
module load python/3.5
python /home/scott/jamesd/Bandit_RandEqual_Cst1/runFFRBandit.py 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=BT_6
#SBATCH --error=/work/scott/jamesd/Bandit_RandEqual_Cst1/log/job.%J.BT_6.err
#SBATCH --output=/work/scott/jamesd/Bandit_RandEqual_Cst1/log/job.%J.BT_6.out
module load python/3.5
python /home/scott/jamesd/Bandit_RandEqual_Cst1/runFFRBandit.py 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=BT_7
#SBATCH --error=/work/scott/jamesd/Bandit_RandEqual_Cst1/log/job.%J.BT_7.err
#SBATCH --output=/work/scott/jamesd/Bandit_RandEqual_Cst1/log/job.%J.BT_7.out
module load python/3.5
python /home/scott/jamesd/Bandit_RandEqual_Cst1/runFFRBandit.py 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=BT_8
#SBATCH --error=/work/scott/jamesd/Bandit_RandEqual_Cst1/log/job.%J.BT_8.err
#SBATCH --output=/work/scott/jamesd/Bandit_RandEqual_Cst1/log/job.%J.BT_8.out
module load python/3.5
python /home/scott/jamesd/Bandit_RandEqual_Cst1/runFFRBandit.py 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=BT_9
#SBATCH --error=/work/scott/jamesd/Bandit_RandEqual_Cst1/log/job.%J.BT_9.err
#SBATCH --output=/work/scott/jamesd/Bandit_RandEqual_Cst1/log/job.%J.BT_9.out
module load python/3.5
python /home/scott/jamesd/Bandit_RandEqual_Cst1/runFFRBandit.py 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=BT_10
#SBATCH --error=/work/scott/jamesd/Bandit_RandEqual_Cst1/log/job.%J.BT_10.err
#SBATCH --output=/work/scott/jamesd/Bandit_RandEqual_Cst1/log/job.%J.BT_10.out
module load python/3.5
python /home/scott/jamesd/Bandit_RandEqual_Cst1/runFFRBandit.py 10
EOF

