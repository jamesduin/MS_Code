#!/usr/bin/env bash

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=10:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=F0p6_3
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst2/log/job.%J.FFR_0p6_3.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst2/log/job.%J.FFR_0p6_3.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst2/runFFR.py 0.6 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=10:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p6_4
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst2/log/job.%J.FFR_0p6_4.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst2/log/job.%J.FFR_0p6_4.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst2/runFFR.py 0.6 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=10:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=F0p7_3
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst2/log/job.%J.FFR_0p7_3.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst2/log/job.%J.FFR_0p7_3.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst2/runFFR.py 0.7 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=10:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p7_4
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst2/log/job.%J.FFR_0p7_4.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst2/log/job.%J.FFR_0p7_4.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst2/runFFR.py 0.7 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=10:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=F0p8_3
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst2/log/job.%J.FFR_0p8_3.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst2/log/job.%J.FFR_0p8_3.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst2/runFFR.py 0.8 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=10:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p8_4
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst2/log/job.%J.FFR_0p8_4.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst2/log/job.%J.FFR_0p8_4.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst2/runFFR.py 0.8 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=10:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=F0p9_3
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst2/log/job.%J.FFR_0p9_3.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst2/log/job.%J.FFR_0p9_3.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst2/runFFR.py 0.9 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=10:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F0p9_4
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst2/log/job.%J.FFR_0p9_4.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst2/log/job.%J.FFR_0p9_4.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst2/runFFR.py 0.9 4
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=10:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu_m2070
#SBATCH --job-name=F1p0_3
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst2/log/job.%J.FFR_1p0_3.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst2/log/job.%J.FFR_1p0_3.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst2/runFFR.py 1.0 3
EOF

sbatch <<'EOF'
#!/bin/sh
#SBATCH --time=10:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1       # number of nodes
#SBATCH --ntasks=1       # number of cores
#SBATCH --mem-per-cpu=2024       # Maximum memory required per CPU (in megabytes)
#SBATCH --job-name=F1p0_4
#SBATCH --error=/work/scott/jamesd/runFFRR_Cst2/log/job.%J.FFR_1p0_4.err
#SBATCH --output=/work/scott/jamesd/runFFRR_Cst2/log/job.%J.FFR_1p0_4.out
module load python/3.5
python /home/scott/jamesd/runFFRR_Cst2/runFFR.py 1.0 4
EOF