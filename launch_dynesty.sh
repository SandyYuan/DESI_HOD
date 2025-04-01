#!/bin/bash

#SBATCH --job-name=fulldy
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --qos=regular
#SBATCH -C cpu
#SBATCH -A desi
#SBATCH --mail-user=sandyyuan94@gmail.com
#SBATCH --mail-type=ALL

source /global/homes/s/sihany/anaconda3/etc/profile.d/conda.sh

conda activate perl
module unload craype-hugepages2M

# python run_dynesty_xi_1t.py --path2config config/mock_qso_z1.4.yaml

# # python run_dynesty_xi_1t.py --path2config config/mock_elg_z0.8_fullscale.yaml
srun -N 1 python -u run_dynesty_xi_1t.py --path2config config/mock_lrg_z0.5_2ndgen.yaml & 
# srun -N 1 python -u run_dynesty_wp_1t.py --path2config config/mock_mt_z0.8_wp_1.yaml & 
# srun -N 1 python -u run_dynesty_wp_1t.py --path2config config/mock_mt_z0.8_wp_2.yaml & 
# srun -N 1 python -u run_dynesty_wp_1t.py --path2config config/mock_mt_z0.8_wp_3.yaml & 

wait