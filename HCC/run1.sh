#!/bin/sh

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p0_1
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.0_1.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.0_1.out
#SBATCH --qos=short
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.0 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p0_2
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.0_2.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.0_2.out
#SBATCH --qos=short
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.0 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p0_3
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.0_3.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.0_3.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.0 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p0_4
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.0_4.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.0_4.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.0 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p0_5
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.0_5.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.0_5.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.0 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p0_6
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.0_6.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.0_6.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.0 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p0_7
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.0_7.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.0_7.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.0 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p1_1
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.1_1.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.1_1.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.1 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p1_2
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.1_2.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.1_2.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.1 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p1_3
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.1_3.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.1_3.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.1 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p1_4
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.1_4.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.1_4.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.1 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p1_5
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.1_5.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.1_5.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.1 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p1_6
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.1_6.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.1_6.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.1 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p1_7
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.1_7.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.1_7.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.1 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p2_1
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.2_1.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.2_1.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.2 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p2_2
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.2_2.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.2_2.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.2 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p2_3
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.2_3.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.2_3.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.2 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p2_4
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.2_4.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.2_4.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.2 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p2_5
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.2_5.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.2_5.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.2 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p2_6
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.2_6.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.2_6.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.2 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p2_7
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.2_7.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.2_7.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.2 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p3_1
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.3_1.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.3_1.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.3 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p3_2
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.3_2.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.3_2.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.3 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p3_3
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.3_3.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.3_3.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.3 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p3_4
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.3_4.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.3_4.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.3 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p3_5
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.3_5.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.3_5.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.3 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p3_6
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.3_6.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.3_6.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.3 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p3_7
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.3_7.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.3_7.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.3 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p4_1
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.4_1.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.4_1.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.4 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p4_2
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.4_2.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.4_2.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.4 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p4_3
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.4_3.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.4_3.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.4 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p4_4
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.4_4.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.4_4.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.4 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p4_5
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.4_5.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.4_5.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.4 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p4_6
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.4_6.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.4_6.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.4 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p4_7
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.4_7.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.4_7.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.4 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p5_1
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.5_1.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.5_1.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.5 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p5_2
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.5_2.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.5_2.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.5 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p5_3
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.5_3.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.5_3.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.5 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p5_4
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.5_4.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.5_4.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.5 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p5_5
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.5_5.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.5_5.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.5 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p5_6
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.5_6.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.5_6.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.5 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p5_7
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.5_7.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.5_7.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.5 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p6_1
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.6_1.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.6_1.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.6 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p6_2
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.6_2.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.6_2.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.6 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p6_3
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.6_3.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.6_3.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.6 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p6_4
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.6_4.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.6_4.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.6 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p6_5
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.6_5.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.6_5.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.6 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p6_6
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.6_6.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.6_6.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.6 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p6_7
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.6_7.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.6_7.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.6 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p7_1
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.7_1.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.7_1.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.7 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p7_2
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.7_2.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.7_2.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.7 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p7_3
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.7_3.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.7_3.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.7 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p7_4
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.7_4.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.7_4.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.7 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p7_5
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.7_5.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.7_5.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.7 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p7_6
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.7_6.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.7_6.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.7 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p7_7
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.7_7.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.7_7.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.7 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p8_1
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.8_1.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.8_1.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.8 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p8_2
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.8_2.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.8_2.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.8 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p8_3
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.8_3.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.8_3.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.8 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p8_4
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.8_4.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.8_4.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.8 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p8_5
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.8_5.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.8_5.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.8 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p8_6
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.8_6.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.8_6.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.8 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p8_7
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.8_7.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.8_7.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.8 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p9_1
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.9_1.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.9_1.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.9 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p9_2
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.9_2.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.9_2.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.9 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p9_3
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.9_3.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.9_3.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.9 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p9_4
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.9_4.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.9_4.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.9 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p9_5
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.9_5.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.9_5.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.9 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p9_6
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.9_6.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.9_6.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.9 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p9_7
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.9_7.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_0.9_7.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 0.9 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F1p0_1
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_1.0_1.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_1.0_1.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 1.0 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F1p0_2
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_1.0_2.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_1.0_2.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 1.0 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F1p0_3
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_1.0_3.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_1.0_3.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 1.0 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F1p0_4
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_1.0_4.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_1.0_4.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 1.0 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F1p0_5
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_1.0_5.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_1.0_5.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 1.0 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F1p0_6
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_1.0_6.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_1.0_6.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 1.0 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=1:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F1p0_7
#SBATCH --error=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_1.0_7.err
#SBATCH --output=/work/scott/jamesd/runFFR_Cst16/log/job.%J.FFR_1.0_7.out
module load python/3.5
python /home/scott/jamesd/runFFR_Cst16/runFFR.py 1.0 7
EOF

