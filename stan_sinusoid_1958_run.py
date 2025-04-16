#!/usr/bin/env python

import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
from isoweek import Week
from cmdstanpy import CmdStanModel

# Read the Danish data
df = pd.read_csv("data/DK_1958_1995_pseudofrac.csv", dtype=str)

val_colname = "PseudoFrac"

df[val_colname] = pd.to_numeric(df[val_colname])

df = df.rename(columns={val_colname: "value"})

# Square root transform:
pretransform_mean = df["value"].mean()
df["value"] = np.sqrt(df["value"])
df["value"] = df["value"] * pretransform_mean/df["value"].mean()

df["value"] = pd.to_numeric(df["value"])

df["quarter_consec"] = pd.to_numeric(df["quarter_consec"])

N = df.shape[0]
Npred = 4

# Create the betawhich array
idx_range = np.arange(1, N + 1)
betawhich = np.ceil(idx_range / (4.0 * 4)).astype(int)

# Create the rhowhich array
idx_range = np.arange(1, N + 1)
rhowhich = np.ceil(idx_range / (4.0 * 4)).astype(int)

# Number of different beta levels 
Nbeta = betawhich.max() if len(betawhich) > 0 else 0

# Number of different rho levels 
Nrho = rhowhich.max() if len(rhowhich) > 0 else 0

stan_data = {
    "N": N,
    "Npred": Npred,
    "betawhich": betawhich,
    "rhowhich": rhowhich,
    "quarter": df["quarter_consec"].tolist(),
    "positivity": df["value"].tolist(),
    "mu": 1 / (75.0 * 4.0), 
    "pop": 1.0,
    "T": 2.5/13.0,
    "delta": 13*0.001713,
    "scale_time_step": 4*13,
    "Nbeta": Nbeta,
    "Nrho": Nrho,
}

stan_file = "sirs_1958_betafac_rhofac_sqrt.stan"
sir_model = CmdStanModel(stan_file=stan_file)

n_chains = 4

# Inits
inits = [{'S0': 0.58, 'logx_I0': -2.15, 'beta0': 10.0, 'dbeta': 0.15, 'betaphase': 1.5, 'sigma_obs': 0.10, 'logrho': -0.3}] * n_chains 

import time
time.sleep(2)

fit = sir_model.sample(
    data=stan_data,
    chains=n_chains,
    parallel_chains=n_chains,
    inits=inits,
    iter_sampling=600,
    iter_warmup=400,
    adapt_delta=0.98,
    adapt_engaged=True,
    max_treedepth=12,
    show_console=True,
    refresh=10,
)

print(fit.diagnose())

n_divergent = np.sum(fit.divergences)
print(f"Number of divergent transitions: {n_divergent}")

output_dir = "stan_output/sinusoid_1958/"
os.makedirs(output_dir, exist_ok=True)

fit.save_csvfiles(dir=output_dir)

print(f"Sampling complete. Outputs saved in '{output_dir}'.")
