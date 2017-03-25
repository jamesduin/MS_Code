

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
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
#SBATCH --partition=highmem
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
#SBATCH --partition=highmem
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
#SBATCH --partition=highmem
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
#SBATCH --partition=highmem
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

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p1_1
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p1_1.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p1_1.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.1 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p1_2
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p1_2.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p1_2.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.1 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p1_3
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p1_3.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p1_3.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.1 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p1_4
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p1_4.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p1_4.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.1 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p1_5
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p1_5.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p1_5.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.1 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p1_6
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p1_6.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p1_6.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.1 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p1_7
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p1_7.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p1_7.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.1 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p1_8
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p1_8.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p1_8.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.1 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p1_9
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p1_9.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p1_9.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.1 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p1_10
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p1_10.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p1_10.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.1 10
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p2_1
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p2_1.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p2_1.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.2 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p2_2
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p2_2.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p2_2.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.2 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p2_3
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p2_3.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p2_3.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.2 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p2_4
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p2_4.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p2_4.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.2 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p2_5
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p2_5.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p2_5.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.2 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p2_6
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p2_6.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p2_6.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.2 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p2_7
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p2_7.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p2_7.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.2 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p2_8
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p2_8.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p2_8.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.2 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p2_9
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p2_9.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p2_9.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.2 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p2_10
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p2_10.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p2_10.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.2 10
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p3_1
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p3_1.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p3_1.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.3 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p3_2
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p3_2.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p3_2.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.3 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p3_3
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p3_3.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p3_3.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.3 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p3_4
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p3_4.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p3_4.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.3 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p3_5
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p3_5.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p3_5.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.3 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p3_6
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p3_6.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p3_6.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.3 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p3_7
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p3_7.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p3_7.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.3 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p3_8
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p3_8.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p3_8.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.3 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p3_9
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p3_9.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p3_9.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.3 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p3_10
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p3_10.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p3_10.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.3 10
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p4_1
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p4_1.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p4_1.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.4 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p4_2
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p4_2.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p4_2.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.4 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p4_3
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p4_3.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p4_3.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.4 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p4_4
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p4_4.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p4_4.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.4 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p4_5
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p4_5.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p4_5.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.4 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p4_6
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p4_6.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p4_6.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.4 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p4_7
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p4_7.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p4_7.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.4 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p4_8
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p4_8.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p4_8.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.4 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p4_9
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p4_9.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p4_9.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.4 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p4_10
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p4_10.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p4_10.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.4 10
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p5_1
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p5_1.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p5_1.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.5 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p5_2
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p5_2.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p5_2.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.5 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p5_3
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p5_3.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p5_3.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.5 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p5_4
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p5_4.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p5_4.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.5 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p5_5
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p5_5.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p5_5.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.5 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p5_6
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p5_6.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p5_6.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.5 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p5_7
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p5_7.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p5_7.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.5 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p5_8
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p5_8.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p5_8.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.5 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p5_9
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p5_9.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p5_9.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.5 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p5_10
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p5_10.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p5_10.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.5 10
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p6_1
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p6_1.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p6_1.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.6 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p6_2
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p6_2.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p6_2.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.6 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p6_3
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p6_3.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p6_3.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.6 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p6_4
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p6_4.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p6_4.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.6 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p6_5
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p6_5.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p6_5.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.6 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p6_6
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p6_6.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p6_6.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.6 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p6_7
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p6_7.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p6_7.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.6 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p6_8
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p6_8.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p6_8.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.6 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p6_9
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p6_9.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p6_9.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.6 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p6_10
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p6_10.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p6_10.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.6 10
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p7_1
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p7_1.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p7_1.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.7 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p7_2
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p7_2.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p7_2.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.7 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p7_3
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p7_3.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p7_3.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.7 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p7_4
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p7_4.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p7_4.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.7 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p7_5
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p7_5.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p7_5.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.7 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p7_6
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p7_6.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p7_6.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.7 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p7_7
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p7_7.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p7_7.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.7 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p7_8
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p7_8.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p7_8.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.7 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p7_9
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p7_9.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p7_9.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.7 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p7_10
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p7_10.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p7_10.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.7 10
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p8_1
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p8_1.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p8_1.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.8 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p8_2
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p8_2.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p8_2.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.8 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p8_3
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p8_3.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p8_3.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.8 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p8_4
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p8_4.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p8_4.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.8 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p8_5
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p8_5.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p8_5.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.8 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p8_6
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p8_6.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p8_6.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.8 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p8_7
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p8_7.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p8_7.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.8 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p8_8
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p8_8.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p8_8.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.8 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p8_9
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p8_9.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p8_9.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.8 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p8_10
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p8_10.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p8_10.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.8 10
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p9_1
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p9_1.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p9_1.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.9 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p9_2
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p9_2.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p9_2.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.9 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p9_3
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p9_3.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p9_3.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.9 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p9_4
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p9_4.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p9_4.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.9 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F0p9_5
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p9_5.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p9_5.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.9 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p9_6
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p9_6.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p9_6.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.9 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p9_7
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p9_7.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p9_7.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.9 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p9_8
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p9_8.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p9_8.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.9 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p9_9
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p9_9.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p9_9.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.9 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p9_10
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p9_10.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_0p9_10.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 0.9 10
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F1p0_1
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_1p0_1.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_1p0_1.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 1.0 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F1p0_2
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_1p0_2.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_1p0_2.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 1.0 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F1p0_3
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_1p0_3.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_1p0_3.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 1.0 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F1p0_4
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_1p0_4.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_1p0_4.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 1.0 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=highmem
#SBATCH --job-name=F1p0_5
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_1p0_5.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_1p0_5.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 1.0 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F1p0_6
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_1p0_6.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_1p0_6.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 1.0 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F1p0_7
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_1p0_7.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_1p0_7.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 1.0 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F1p0_8
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_1p0_8.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_1p0_8.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 1.0 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F1p0_9
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_1p0_9.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_1p0_9.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 1.0 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=20:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F1p0_10
#SBATCH --error=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_1p0_10.err
#SBATCH --output=/work/scott/jamesd/runFFRParam_1p2/log/job.%J.FFR_1p0_10.out
module load python/3.5
python /home/scott/jamesd/FFRParam/runFFRParam.py runFFRParam 1.2 1.0 10
EOF

