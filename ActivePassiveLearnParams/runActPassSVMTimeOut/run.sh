#!/bin/sh

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=AS_a_1
#SBATCH --error=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_active_1.err
#SBATCH --output=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_active_1.out
#SBATCH --qos=short
module load python/3.5
python /home/scott/jamesd/runActPassParam/runActPassParam.py SVM runActPassSVM active 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=AS_a_2
#SBATCH --error=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_active_2.err
#SBATCH --output=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_active_2.out
#SBATCH --qos=short
module load python/3.5
python /home/scott/jamesd/runActPassParam/runActPassParam.py SVM runActPassSVM active 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_k20
#SBATCH --job-name=AS_a_3
#SBATCH --error=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_active_3.err
#SBATCH --output=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_active_3.out
module load python/3.5
python /home/scott/jamesd/runActPassParam/runActPassParam.py SVM runActPassSVM active 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_k20
#SBATCH --job-name=AS_a_4
#SBATCH --error=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_active_4.err
#SBATCH --output=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_active_4.out
module load python/3.5
python /home/scott/jamesd/runActPassParam/runActPassParam.py SVM runActPassSVM active 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=AS_a_5
#SBATCH --error=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_active_5.err
#SBATCH --output=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_active_5.out
module load python/3.5
python /home/scott/jamesd/runActPassParam/runActPassParam.py SVM runActPassSVM active 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=AS_a_6
#SBATCH --error=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_active_6.err
#SBATCH --output=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_active_6.out
module load python/3.5
python /home/scott/jamesd/runActPassParam/runActPassParam.py SVM runActPassSVM active 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=AS_a_7
#SBATCH --error=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_active_7.err
#SBATCH --output=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_active_7.out
module load python/3.5
python /home/scott/jamesd/runActPassParam/runActPassParam.py SVM runActPassSVM active 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=AS_a_8
#SBATCH --error=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_active_8.err
#SBATCH --output=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_active_8.out
module load python/3.5
python /home/scott/jamesd/runActPassParam/runActPassParam.py SVM runActPassSVM active 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=AS_a_9
#SBATCH --error=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_active_9.err
#SBATCH --output=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_active_9.out
module load python/3.5
python /home/scott/jamesd/runActPassParam/runActPassParam.py SVM runActPassSVM active 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=AS_a_10
#SBATCH --error=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_active_10.err
#SBATCH --output=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_active_10.out
module load python/3.5
python /home/scott/jamesd/runActPassParam/runActPassParam.py SVM runActPassSVM active 10
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=AS_p_1
#SBATCH --error=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_passive_1.err
#SBATCH --output=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_passive_1.out
module load python/3.5
python /home/scott/jamesd/runActPassParam/runActPassParam.py SVM runActPassSVM passive 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=AS_p_2
#SBATCH --error=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_passive_2.err
#SBATCH --output=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_passive_2.out
module load python/3.5
python /home/scott/jamesd/runActPassParam/runActPassParam.py SVM runActPassSVM passive 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_k20
#SBATCH --job-name=AS_p_3
#SBATCH --error=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_passive_3.err
#SBATCH --output=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_passive_3.out
module load python/3.5
python /home/scott/jamesd/runActPassParam/runActPassParam.py SVM runActPassSVM passive 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_k20
#SBATCH --job-name=AS_p_4
#SBATCH --error=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_passive_4.err
#SBATCH --output=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_passive_4.out
module load python/3.5
python /home/scott/jamesd/runActPassParam/runActPassParam.py SVM runActPassSVM passive 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=AS_p_5
#SBATCH --error=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_passive_5.err
#SBATCH --output=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_passive_5.out
module load python/3.5
python /home/scott/jamesd/runActPassParam/runActPassParam.py SVM runActPassSVM passive 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=AS_p_6
#SBATCH --error=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_passive_6.err
#SBATCH --output=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_passive_6.out
module load python/3.5
python /home/scott/jamesd/runActPassParam/runActPassParam.py SVM runActPassSVM passive 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=AS_p_7
#SBATCH --error=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_passive_7.err
#SBATCH --output=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_passive_7.out
module load python/3.5
python /home/scott/jamesd/runActPassParam/runActPassParam.py SVM runActPassSVM passive 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=AS_p_8
#SBATCH --error=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_passive_8.err
#SBATCH --output=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_passive_8.out
module load python/3.5
python /home/scott/jamesd/runActPassParam/runActPassParam.py SVM runActPassSVM passive 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=AS_p_9
#SBATCH --error=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_passive_9.err
#SBATCH --output=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_passive_9.out
module load python/3.5
python /home/scott/jamesd/runActPassParam/runActPassParam.py SVM runActPassSVM passive 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=3:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=AS_p_10
#SBATCH --error=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_passive_10.err
#SBATCH --output=/work/scott/jamesd/runActPassSVM/log/job.%J.AP_passive_10.out
module load python/3.5
python /home/scott/jamesd/runActPassParam/runActPassParam.py SVM runActPassSVM passive 10
EOF

