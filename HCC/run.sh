#!/bin/sh

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=15:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=FFR_0_0
#SBATCH --error=/work/scott/jamesd/resultsFFR_1/log/job.%J.FFR_0_0.err
#SBATCH --output=/work/scott/jamesd/resultsFFR_1/log/job.%J.FFR_0_0.out
module load python/3.5
python /home/scott/jamesd/resultsFFR_1/runFFR.py 0.0
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=15:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=FFR_0_1
#SBATCH --error=/work/scott/jamesd/resultsFFR_1/log/job.%J.FFR_0_1.err
#SBATCH --output=/work/scott/jamesd/resultsFFR_1/log/job.%J.FFR_0_1.out
module load python/3.5
python /home/scott/jamesd/resultsFFR_1/runFFR.py 0.1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=15:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=FFR_0_2
#SBATCH --error=/work/scott/jamesd/resultsFFR_1/log/job.%J.FFR_0_2.err
#SBATCH --output=/work/scott/jamesd/resultsFFR_1/log/job.%J.FFR_0_2.out
module load python/3.5
python /home/scott/jamesd/resultsFFR_1/runFFR.py 0.2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=15:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=FFR_0_3
#SBATCH --error=/work/scott/jamesd/resultsFFR_1/log/job.%J.FFR_0_3.err
#SBATCH --output=/work/scott/jamesd/resultsFFR_1/log/job.%J.FFR_0_3.out
module load python/3.5
python /home/scott/jamesd/resultsFFR_1/runFFR.py 0.3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=15:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=FFR_0_4
#SBATCH --error=/work/scott/jamesd/resultsFFR_1/log/job.%J.FFR_0_4.err
#SBATCH --output=/work/scott/jamesd/resultsFFR_1/log/job.%J.FFR_0_4.out
module load python/3.5
python /home/scott/jamesd/resultsFFR_1/runFFR.py 0.4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=15:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=FFR_0_5
#SBATCH --error=/work/scott/jamesd/resultsFFR_1/log/job.%J.FFR_0_5.err
#SBATCH --output=/work/scott/jamesd/resultsFFR_1/log/job.%J.FFR_0_5.out
module load python/3.5
python /home/scott/jamesd/resultsFFR_1/runFFR.py 0.5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=15:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=FFR_0_6
#SBATCH --error=/work/scott/jamesd/resultsFFR_1/log/job.%J.FFR_0_6.err
#SBATCH --output=/work/scott/jamesd/resultsFFR_1/log/job.%J.FFR_0_6.out
module load python/3.5
python /home/scott/jamesd/resultsFFR_1/runFFR.py 0.6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=15:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=FFR_0_7
#SBATCH --error=/work/scott/jamesd/resultsFFR_1/log/job.%J.FFR_0_7.err
#SBATCH --output=/work/scott/jamesd/resultsFFR_1/log/job.%J.FFR_0_7.out
module load python/3.5
python /home/scott/jamesd/resultsFFR_1/runFFR.py 0.7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=15:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=FFR_0_8
#SBATCH --error=/work/scott/jamesd/resultsFFR_1/log/job.%J.FFR_0_8.err
#SBATCH --output=/work/scott/jamesd/resultsFFR_1/log/job.%J.FFR_0_8.out
module load python/3.5
python /home/scott/jamesd/resultsFFR_1/runFFR.py 0.8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=15:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=FFR_0_9
#SBATCH --error=/work/scott/jamesd/resultsFFR_1/log/job.%J.FFR_0_9.err
#SBATCH --output=/work/scott/jamesd/resultsFFR_1/log/job.%J.FFR_0_9.out
module load python/3.5
python /home/scott/jamesd/resultsFFR_1/runFFR.py 0.9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=15:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=FFR_1_0
#SBATCH --error=/work/scott/jamesd/resultsFFR_1/log/job.%J.FFR_1_0.err
#SBATCH --output=/work/scott/jamesd/resultsFFR_1/log/job.%J.FFR_1_0.out
module load python/3.5
python /home/scott/jamesd/resultsFFR_1/runFFR.py 1.0
EOF

