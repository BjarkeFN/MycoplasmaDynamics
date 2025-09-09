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

# Number of observed data points
N = df.shape[0]

# Number of prediction time points
Npred = 4

idx_range = np.arange(1, N + 1)
betawhich = np.ceil(idx_range / (4.0 * 4)).astype(int)

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
    "mu": 1 / (75.0 * 4.0),  # 1 / (80*52)
    "pop": 1.0,
    "T": 2.5/13.0,
    "delta": 13*0.0023089,
    "scale_time_step": 8*13,
    "Nbeta": Nbeta,
    "Nrho": Nrho,
    #"rho": 1.0/57.0,
}


stan_file = "stan_sinusoid_1958.stan"

sir_model = CmdStanModel(stan_file=stan_file)

n_chains = 4

# Inits
inits = [{'S0': 0.58, 'logx_I0': -2.15, 'beta0': 8.0, 'dbeta': 0.15, 'betaphase': 1.5, 'sigma_obs': 0.10, 'logrho': -0.3}] * n_chains 

print(f"Data points: {len(df['value'])}")
print(f"Nbeta: {Nbeta}. betawhich: {betawhich}")
print(f"len(betawhich): {len(betawhich)}")

import time
time.sleep(2)

fit = sir_model.sample(
    data=stan_data,
    chains=n_chains,
    parallel_chains=n_chains,
    inits=inits,
    iter_sampling=1500,
    iter_warmup=1500,  # total iter = iter_warmup + iter_sampling
    adapt_delta=0.98,
    #step_size=0.01,
    adapt_engaged=True,
    max_treedepth=12,
    show_console=True,
    refresh=10,
)

print(fit.diagnose())

n_divergent = np.sum(fit.divergences)  # CmdStan's 'divergent__' is in column 0
print(f"Number of divergent transitions: {n_divergent}")

output_dir = "stan_output/sinusoid_1958/"
os.makedirs(output_dir, exist_ok=True)

fit.save_csvfiles(dir=output_dir)

summary_df = fit.summary()
summary_path = os.path.join(output_dir, "stanfit_sirs_synth_summary.csv")
summary_df.to_csv(summary_path)

print(f"Sampling complete. Diagnostics and summary saved in '{output_dir}'.")
