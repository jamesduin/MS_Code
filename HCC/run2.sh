#!/bin/sh

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_k20
#SBATCH --job-name=F0p9_1
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_0.9_1.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_0.9_1.out
#SBATCH --qos=short
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst16/runFFR.py 0.9 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_k20
#SBATCH --job-name=F0p9_2
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_0.9_2.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_0.9_2.out
#SBATCH --qos=short
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst16/runFFR.py 0.9 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=F0p9_3
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_0.9_3.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_0.9_3.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst16/runFFR.py 0.9 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=F0p9_4
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_0.9_4.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_0.9_4.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst16/runFFR.py 0.9 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=F0p9_5
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_0.9_5.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_0.9_5.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst16/runFFR.py 0.9 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=F0p9_6
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_0.9_6.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_0.9_6.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst16/runFFR.py 0.9 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p9_7
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_0.9_7.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_0.9_7.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst16/runFFR.py 0.9 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p9_8
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_0.9_8.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_0.9_8.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst16/runFFR.py 0.9 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p9_9
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_0.9_9.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_0.9_9.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst16/runFFR.py 0.9 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p9_10
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_0.9_10.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_0.9_10.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst16/runFFR.py 0.9 10
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_k20
#SBATCH --job-name=F1p0_1
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_1.0_1.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_1.0_1.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst16/runFFR.py 1.0 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_k20
#SBATCH --job-name=F1p0_2
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_1.0_2.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_1.0_2.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst16/runFFR.py 1.0 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=F1p0_3
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_1.0_3.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_1.0_3.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst16/runFFR.py 1.0 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=F1p0_4
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_1.0_4.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_1.0_4.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst16/runFFR.py 1.0 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=F1p0_5
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_1.0_5.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_1.0_5.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst16/runFFR.py 1.0 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=F1p0_6
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_1.0_6.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_1.0_6.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst16/runFFR.py 1.0 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F1p0_7
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_1.0_7.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_1.0_7.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst16/runFFR.py 1.0 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F1p0_8
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_1.0_8.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_1.0_8.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst16/runFFR.py 1.0 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F1p0_9
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_1.0_9.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_1.0_9.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst16/runFFR.py 1.0 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=6:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F1p0_10
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_1.0_10.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst16/log/job.%J.FFR_1.0_10.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst16/runFFR.py 1.0 10
EOF

