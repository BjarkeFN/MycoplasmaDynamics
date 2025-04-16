data {
  int N; 
  int Npred;
  int Nnonpi;
  int Nnpi;
  int<lower=1> scale_time_step; // Number of simulated timesteps per data timestep
  array[N - Nnonpi] int npiwhich;
  array[N + Npred] int week;
  array[N] real positivity; // Observed positivity
  real mu;
  real pop;
  real<lower=0> T;        // Infectious period
  real<lower=0> delta; // Immune waning
}

parameters {
  real<lower=0.4, upper=0.65> S0;
  real<lower=-7, upper=0> logx_I0; // For stick-breaking method
  real<lower=-2, upper=1> logrho;
  real<lower=0.0, upper=0.5> sigma;
  array[52] real<lower=0.0, upper=5.0> beta;
  real<lower=0> sigma_obs;
  array[Nnpi] real<lower=0> npi;
}

transformed parameters {
  real<lower=0, upper=10> rho = pow(10, logrho);
  real<lower=0, upper=1> I0 = S0 * pow(10, logx_I0);
  real<lower=0> gamma = 1.0 / T;
  vector<lower=0>[N + Npred] S;
  vector<lower=0>[N + Npred] I;
  vector<lower=0>[N + Npred] R;
  vector<lower=0>[N + Npred] Ifit;
  vector<lower=0>[N + Npred] npieff;
  vector<lower=0>[N + Npred] inc;
  
  array[N + Npred] real times;
  for (i in 1:N + Npred) {
    times[i] = i * 1.0;
  }
  
  // Initial conditions
  S[1] = S0 * pop;
  I[1] = I0 * pop;
  R[1] = (1 - S0 - I0) * pop;

  // Assign NPI effects
  for (i in 1:Nnonpi) {
    npieff[i] = 1;
  }
  for (i in (Nnonpi + 1):N) {
    npieff[i] = npi[npiwhich[i - Nnonpi]];
  }
  for (i in (N + 1):(N + Npred)) {
    npieff[i] = 1;
  }

  // SIRS model with subdivided timesteps
  for (i in 2:(N + Npred)) {
    real dt = 1.0/scale_time_step;
    vector[scale_time_step+1] Ssub;
    vector[scale_time_step+1] Isub;
    vector[scale_time_step+1] Rsub;
    vector[scale_time_step+1] incsub;
    Ssub[1] = S[i-1];
    Isub[1] = I[i-1];
    Rsub[1] = R[i-1];
    incsub[1] = inc[i-1];
    int mi0 = i * scale_time_step + 1;
    int mi_end = mi0+scale_time_step-1;
    for (mi in mi0:mi_end) {
        int mi_loc = mi-mi0+1;
        real foi  = npieff[i] * beta[week[i]] * Isub[mi_loc] / pop; // Force of infection
        real Sout = (1 - exp(-(foi + mu)*dt)) * Ssub[mi_loc];         // Susceptibles leaving S
        real StoI = foi/(foi + mu) * Sout;                   // Transition from S to I
        real Iout = (1 - exp(-(gamma + mu)*dt)) * Isub[mi_loc];       // Infectious leaving I
        real ItoR = gamma / (gamma + mu) * Iout;               // Transition from I to R
        real Rout = (1 - exp(-(delta + mu)*dt)) * Rsub[mi_loc];       // Recovered leaving R
        real RtoS = delta / (delta + mu) * Rout;               // Transition from R to S (waning immunity)
        Ssub[mi_loc+1] = Ssub[mi_loc] - Sout + mu * pop * dt + RtoS;              // Update S with immune waning
        Isub[mi_loc+1] = Isub[mi_loc] + StoI - Iout;                         // Update I
        Rsub[mi_loc+1] = Rsub[mi_loc] + ItoR - Rout;                         // Update R
        incsub[mi_loc+1]=StoI;
        if (mi == mi_end) {
            S[i] = Ssub[mi_loc+1];
            I[i] = Isub[mi_loc+1];
            R[i] = Rsub[mi_loc+1];
            inc[i] = sum(incsub[2:]);
            //print("inc[", i, "] = ", inc[i]);
        }
    }  
  }
  inc[1]=inc[2]; // Since we have no better estimate of inc[1]

  // Predicted observed cases
  for (i in 1:(N + Npred)) {
    //Ifit[i] = I[i] / rho; // Prevalence
    Ifit[i] = inc[i] / rho; // Incidence
  }
}

model {
  // Priors for transmission rates
  sigma ~ normal(0, 0.1);
  beta ~ normal(0.7, 0.2);
  for (i in 2:52) {
    beta[i] ~ normal(beta[i - 1], sigma);
  }
  beta[1] ~ normal(beta[52], sigma); // Circularity in seasonality
  // Priors for NPI effects
  npi ~ normal(0.8, 0.3);

  logrho ~ normal(-2, 0.5);
  sigma_obs ~ normal(0, 0.1);
  S0 ~ normal(0.3, 0.3);
  logx_I0 ~ normal(-4, 2);
  
  // Likelihood for observed data
  for (i in 1:N) {
    if (positivity[i] > 0) {
      positivity[i] ~ normal(Ifit[i], sigma_obs);
    }
  }
}

generated quantities {
  print("I0=", I0, " S0=", S0, " sigma_obs=", sigma_obs, " rho=", rho, " beta[1]=", beta[1]);
}
