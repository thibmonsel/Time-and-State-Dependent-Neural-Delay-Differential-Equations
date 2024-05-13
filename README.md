# ECAI_Submission

We provide instructions to generate all the data used in our paper along with the files to train each model and the papers additional experiments.

# To generate generate data for ... 

## ... the Time Dependent datasets : 

```bash 
python generate_dataset.py --dataset=time_dependent --noise_level=0
python generate_dataset.py --dataset=time_dependent --noise_level=2
python generate_dataset.py --dataset=time_dependent --noise_level=5
python generate_dataset.py --dataset=time_dependent --noise_level=10
```

## ... the State Dependent datasets : 

```bash 
python generate_dataset.py --dataset=state_dependent 
```

## ... the Diffusion Delay datasets : 

```bash 
python generate_dataset.py --dataset=diffusion_delay 
```

# To launch training for experiments :

You must install diffrax locally by : 
```bash
cd diffrax/
pip install . 
cd ..
```

To launch the training you must refer the `seed` and the model (e.g. `["anode", "ode", "dde", "latent_ode"]`) used along with the noise level (only for the Time Dependent dataset)

## For the Time Dependent dataset with a certain noise level : 

```bash 
python time_dependent.py --model=MODEL --exp_path=EXP_PATH --seed=SEED --noise_level=NOISE_LEVEL
```

## For the State Dependent dataset : 

```bash 
python state_dependent.py --model=MODEL --exp_path=EXP_PATH --seed=SEED
```

## For the Diffusion Delay dataset : 

```bash 
python state_dependent.py --model=MODEL --exp_path=EXP_PATH --seed=SEED
```

## To plot additional experiments

*To plots the other experiments, you must train some model and change the path variables at the start of the file.*

For the extrapolation regime : 
```bash 
python extrapolation_experiments.py --dataset=DATASET
```

For the other history function 
```bash 
python different_history_experiments.py --dataset=DATASET
```
