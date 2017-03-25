

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p0_1
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p0_1.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p0_1.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.0 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p0_2
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p0_2.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p0_2.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.0 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p0_3
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p0_3.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p0_3.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.0 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p0_4
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p0_4.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p0_4.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.0 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p0_5
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p0_5.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p0_5.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.0 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p0_6
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p0_6.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p0_6.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.0 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p0_7
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p0_7.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p0_7.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.0 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p0_8
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p0_8.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p0_8.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.0 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p0_9
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p0_9.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p0_9.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.0 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p0_10
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p0_10.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p0_10.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.0 10
EOF

