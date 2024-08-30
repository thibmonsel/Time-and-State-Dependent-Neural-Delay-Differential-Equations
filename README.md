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

Please note that all data we be saved in `data/` path. By default we create 5 different train/test splits for the generate DATASET. 
This can be changed with the argument `--seed_train_test_split` in the launch command above.

# To launch training for experiments :

You must install the repository to install the dependencies : 
```bash
pip install -e . 
```

To launch the training you must refer the `seed`, the model (e.g. `["anode", "ode", "dde", "latent_ode"]`) used along with the noise level (only for the Time Dependent dataset) . Optionally you can specify which train test split to use `seed_train_test_split` in the bash script `launch.sh` to train models.

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

Alternatively, you can the launch script to train the model. 

```bash 
bash launch 1
```

where `1` corresponds to the first train test split. 

## For Neural Laplace model ...

... you much run a seperate file : 

```bash
cd NeuralLaplace/
python dde_integration.py --noise_level=NOISE_LEVEL --dataset=DATASET                                                                                                        (ecai_test) 
```

where `DATASET = ["state_dependent", "time_dependent", "diffusion_delay"]`

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

**Raise an issue if a bug arises**