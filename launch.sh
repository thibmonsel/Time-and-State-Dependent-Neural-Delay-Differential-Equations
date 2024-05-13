#!/bin/sh
#SBATCH --job-name=dde_experiments     
#SBATCH --time=3-0            
#SBATCH --partition=besteffort
#SBATCH --gres=gpu:1                    
#SBATCH --output=out%j.out              
#SBATCH --error=out%j.err               
set -x


seed_train_test_split=$1
export seed_train_test_split=$1

# python time_dependent.py --model="dde" --exp_path="time_dep/dde" --noise_level=0
# python time_dependent.py --model="ode" --exp_path="time_dep/ode" --noise_level=0
# python time_dependent.py --model="latent_ode" --exp_path="time_dep/latent_ode"  --noise_level=0
# python time_dependent.py --model="anode" --exp_path="time_dep/anode" --noise_level=0

# python state_dependent.py --model="dde" --exp_path="state_dep/dde" 
# python state_dependent.py --model="ode" --exp_path="state_dep/ode" 
# python state_dependent.py --model="latent_ode" --exp_path="state_dep/latent_ode" 
# python state_dependent.py --model="anode" --exp_path="state_dep/anode" 

# python diffusion_delay.py --model="dde" --exp_path="diffusion/dde" 
# python diffusion_delay.py --model="latent_ode" --exp_path="diffusion/latent_ode" 
python diffusion_delay.py --model="anode" --exp_path="diffusion/anode" 
# python diffusion_delay.py --model="ode" --exp_path="diffusion/ode" 

