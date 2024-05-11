#!/bin/sh
#SBATCH --job-name=dde_experiments     
#SBATCH --time=3-0                 
#SBATCH --gres=gpu:1                    
#SBATCH --output=out%j.out              
#SBATCH --error=out%j.err               
set -x


seed_train_test_split=$1
export seed_train_test_split=$1

python diffusion_delay.py --model="latent_ode" --exp_path="diffusion" --seed=1
# python state_dependent.py --model="dde" --exp_path="state_dep" --seed=2
# python time_dependent.py --model="dde" --exp_path="time_dep" --seed=3 --noise_level=2


