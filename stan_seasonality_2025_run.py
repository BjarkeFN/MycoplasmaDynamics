#!/usr/bin/env python

import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
from isoweek import Week
from cmdstanpy import CmdStanModel

# Read the Danish data
df = pd.read_csv("data/DK_2010_2025_posfrac.csv", dtype=str)

val_colname = "PosFrac"

df[val_colname] = pd.to_numeric(df[val_colname])

# parse ISO week format (e.g. "2020-W12") into date (Monday of that ISO week)
def iso_week_to_date(iso_str):
    """
    Convert 'YYYY-Www' string into a Python date.
    Example: "2020-W12" -> Monday of the 12th ISO week of 2020.
    """
    match = re.match(r"(\d{4})-W(\d{1,2})", iso_str)
    if not match:
        return None
    year, week = match.groups()
    year, week = int(year), int(week)
    return Week(year, week).monday()

# Create a date column from year_Week
df["date"] = df["year_Week"].apply(iso_week_to_date)

df['date'] = pd.to_datetime(df["date"])

df = df[df['date'] < "2020-01-01"]

print(df['date'])


df = df.rename(columns={val_colname: "value"})
df["value"] = pd.to_numeric(df["value"])

df["year"] = df["date"].dt.year
df["week"] = df["date"].dt.isocalendar().week.astype(int)

df.loc[df["week"] == 53, "week"] = 52

df = df.sort_values(by="date")
df = df.iloc[:-1]

N = df.shape[0]
Npred = 52 * 10

# define a start time for NPIs in terms of decimal year
npi_start_year = 2020
npi_start_week = 12
npi_start_decimal = npi_start_year + npi_start_week / 52.0

# decimal year + (week/52)
df["decimal_year_week"] = df["year"] + df["week"] / 52.0

Nnonpi = df[df["decimal_year_week"] < npi_start_decimal].shape[0]

if N > Nnonpi:
    idx_range = np.arange(Nnonpi + 1, N + 1)
    npiwhich = np.ceil((idx_range - Nnonpi) / 4.0).astype(int)
else:
    npiwhich = np.array([], dtype=int)

# Number of  NPI levels
Nnpi = npiwhich.max() if len(npiwhich) > 0 else 0

week_vector_length = N + Npred - 1
first_week = df["week"].iloc[0]

repeated_weeks = []
while len(repeated_weeks) < week_vector_length:
    repeated_weeks.extend(range(1, 53))

repeated_weeks = repeated_weeks[:week_vector_length]
week_stan = [first_week] + repeated_weeks


stan_data = {
    "N": N,
    "Npred": Npred,
    "Nnonpi": Nnonpi,
    "Nnpi": Nnpi,
    "npiwhich": npiwhich.tolist() if len(npiwhich) > 0 else [], 
    "week": week_stan,
    "positivity": df["value"].tolist(),
    "mu": 1 / (80.0 * 52.0),  # 1 / (80*52)
    "pop": 1.0,
    "T": 2.5,
	"delta": 0.00172,
    "scale_time_step": 4,
}


stan_file = "stan_seasonality_2025.stan"

sir_model = CmdStanModel(stan_file=stan_file)

n_chains = 4
inits = [{'S0': 0.5, 'logx_I0': -2.5, 'beta': [0.8]*52, 'sigma_obs': 0.10, 'logrho': -1.7}] * n_chains  


fit = sir_model.sample(
    data=stan_data,
    seed=102,
    chains=n_chains,
    parallel_chains=n_chains,
    inits=inits,
    iter_sampling=600,
    iter_warmup=400,
    adapt_delta=0.98,
    max_treedepth=12,
    show_console=True,
    refresh=10
)

print(fit.diagnose())

n_divergent = np.sum(fit.divergences)  
print(f"Number of divergent transitions: {n_divergent}")

output_dir = "stan_output/estimate_seasonality_2025/"
os.makedirs(output_dir, exist_ok=True)

fit.save_csvfiles(dir=output_dir)

print(f"Sampling complete. Outputs saved in '{output_dir}'.")
