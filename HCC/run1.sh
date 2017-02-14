#!/bin/sh

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=8:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=guest
#SBATCH --job-name=AP_p_1
#SBATCH --error=/work/scott/jamesd/resultsRBF1_15RescaleSep/log/job.%J.AP_passive_1.err
#SBATCH --output=/work/scott/jamesd/resultsRBF1_15RescaleSep/log/job.%J.AP_passive_1.out
module load python/3.4
python /home/scott/jamesd/resultsRBF1_15RescaleSep/runActPassRBF.py passive 1
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=8:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=guest
#SBATCH --job-name=AP_p_2
#SBATCH --error=/work/scott/jamesd/resultsRBF1_15RescaleSep/log/job.%J.AP_passive_2.err
#SBATCH --output=/work/scott/jamesd/resultsRBF1_15RescaleSep/log/job.%J.AP_passive_2.out
module load python/3.4
python /home/scott/jamesd/resultsRBF1_15RescaleSep/runActPassRBF.py passive 2
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=8:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=guest
#SBATCH --job-name=AP_p_3
#SBATCH --error=/work/scott/jamesd/resultsRBF1_15RescaleSep/log/job.%J.AP_passive_3.err
#SBATCH --output=/work/scott/jamesd/resultsRBF1_15RescaleSep/log/job.%J.AP_passive_3.out
module load python/3.4
python /home/scott/jamesd/resultsRBF1_15RescaleSep/runActPassRBF.py passive 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=8:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=guest
#SBATCH --job-name=AP_p_4
#SBATCH --error=/work/scott/jamesd/resultsRBF1_15RescaleSep/log/job.%J.AP_passive_4.err
#SBATCH --output=/work/scott/jamesd/resultsRBF1_15RescaleSep/log/job.%J.AP_passive_4.out
module load python/3.4
python /home/scott/jamesd/resultsRBF1_15RescaleSep/runActPassRBF.py passive 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=8:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=guest
#SBATCH --job-name=AP_p_5
#SBATCH --error=/work/scott/jamesd/resultsRBF1_15RescaleSep/log/job.%J.AP_passive_5.err
#SBATCH --output=/work/scott/jamesd/resultsRBF1_15RescaleSep/log/job.%J.AP_passive_5.out
module load python/3.4
python /home/scott/jamesd/resultsRBF1_15RescaleSep/runActPassRBF.py passive 5
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=8:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=guest
#SBATCH --job-name=AP_p_6
#SBATCH --error=/work/scott/jamesd/resultsRBF1_15RescaleSep/log/job.%J.AP_passive_6.err
#SBATCH --output=/work/scott/jamesd/resultsRBF1_15RescaleSep/log/job.%J.AP_passive_6.out
module load python/3.4
python /home/scott/jamesd/resultsRBF1_15RescaleSep/runActPassRBF.py passive 6
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=8:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=guest
#SBATCH --job-name=AP_p_7
#SBATCH --error=/work/scott/jamesd/resultsRBF1_15RescaleSep/log/job.%J.AP_passive_7.err
#SBATCH --output=/work/scott/jamesd/resultsRBF1_15RescaleSep/log/job.%J.AP_passive_7.out
module load python/3.4
python /home/scott/jamesd/resultsRBF1_15RescaleSep/runActPassRBF.py passive 7
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=8:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=guest
#SBATCH --job-name=AP_p_8
#SBATCH --error=/work/scott/jamesd/resultsRBF1_15RescaleSep/log/job.%J.AP_passive_8.err
#SBATCH --output=/work/scott/jamesd/resultsRBF1_15RescaleSep/log/job.%J.AP_passive_8.out
module load python/3.4
python /home/scott/jamesd/resultsRBF1_15RescaleSep/runActPassRBF.py passive 8
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=8:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=guest
#SBATCH --job-name=AP_p_9
#SBATCH --error=/work/scott/jamesd/resultsRBF1_15RescaleSep/log/job.%J.AP_passive_9.err
#SBATCH --output=/work/scott/jamesd/resultsRBF1_15RescaleSep/log/job.%J.AP_passive_9.out
module load python/3.4
python /home/scott/jamesd/resultsRBF1_15RescaleSep/runActPassRBF.py passive 9
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=8:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=guest
#SBATCH --job-name=AP_p_10
#SBATCH --error=/work/scott/jamesd/resultsRBF1_15RescaleSep/log/job.%J.AP_passive_10.err
#SBATCH --output=/work/scott/jamesd/resultsRBF1_15RescaleSep/log/job.%J.AP_passive_10.out
module load python/3.4
python /home/scott/jamesd/resultsRBF1_15RescaleSep/runActPassRBF.py passive 10
EOF

