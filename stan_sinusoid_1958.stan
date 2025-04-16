data {
  int<lower=1> N;         // number of observed quarters
  int<lower=1> Npred;     // number of prediction quarters
  int<lower=1> scale_time_step; // Number of simulated timesteps per data timestep
  int Nbeta;
  int Nrho;
  array[N] real positivity;  // observed positivity (e.g., cases)
  array[N] int betawhich;
  array[N] int rhowhich;
  real<lower=0> mu;       // background birth/death rate
  real<lower=0> pop;      // total population size
  real<lower=0> T;        // Infectious period
  real<lower=0> delta;    // Waning rate
}

parameters {
  real<lower=0.3, upper=0.7> S0;
  real<lower=-7, upper=0> logx_I0; // For stick-breaking method
  // Stick-breaking:
  // Given S0, let I0 = (1-S0)*x_I0. Then let R0=1-(I0+S0)
  
  real<lower=-2, upper=1> logrho;
  
  // Transmission parameters.
  real<lower=1.3/T, upper=2.2/T> beta0;
  real<lower=0, upper=0.3> dbeta;
  real<lower=0, upper=2*pi()> betaphase;
  
  // Observation noise.
  real<lower=0> sigma_obs;
  
  array[Nbeta] real<lower=0.9, upper=1.1> betafac;
  array[Nbeta] real<lower=0.9, upper=1.1> rhofac;
}

transformed parameters {
  real<lower=0, upper=10> rho = pow(10, logrho);
  vector<lower=0>[N + Npred] beff;
  vector<lower=0>[N + Npred] rheff;
  // quarterly time grid (time in quarters: 0, 1, 2, …, N + Npred - 1)
  array[N + Npred] real times;
  for (i in 1:N + Npred) {
    times[i] = i * 1.0;
  }
  
  // Assign effective beta multiplier
  for (i in 1:N) {
    beff[i] = betafac[betawhich[i]];
  }
  for (i in (N + 1):(N + Npred)) {
    beff[i] = 1.0;
  }
  // Assign effective rho multiplier
  for (i in 1:N) {
    rheff[i] = rhofac[rhowhich[i]];
  }
  for (i in (N + 1):(N + Npred)) {
    rheff[i] = 1.0;
  }
  
  // Transmission rate vector:
  vector<lower=0>[scale_time_step * (N + Npred+1)] beta;
  
  array[scale_time_step * (N + Npred+1)] real model_times;
  for (mi in 1:scale_time_step * (N + Npred+1)) {
    model_times[mi] = (mi-1) * 1.0/scale_time_step;
    beta[mi] = beta0 * (1 + dbeta * sin(2 * pi() * model_times[mi] / 4.0 + betaphase));
  }
  
  // Extract the state variables at quarterly time points.
  vector<lower=0, upper=1>[N + Npred] S;
  vector<lower=0, upper=1>[N + Npred] I;
  vector<lower=0, upper=1>[N + Npred] R;
  vector<lower=0>[N + Npred] Ifit;  // model-predicted observed cases (or prevalence)
  vector<lower=0>[N + Npred] inc;
  
  real<lower=0, upper=1> I0 = S0 * pow(10, logx_I0);
  real<lower=0> gamma = 1.0 / T;
  
  // Initial conditions
  S[1] = S0 * pop;
  I[1] = I0 * pop;
  R[1] = (1 - S0 - I0) * pop;


  // SIRS Model Dynamics
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
        real foi  = beff[i] * beta[mi] * Isub[mi_loc] / pop; // Force of infection
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
    Ifit[i] = inc[i] / (rho * rheff[i]); // Incidence
  }
  //print("S, I, R: ", S[1], " ", I[1], " ", R[1]);
}

model {
  // Priors for transmission parameters.
  beta0 ~ normal(0, 5.0/T);
  
  betafac ~ normal(1, 0.05);
  
  rhofac ~ normal(1, 0.20);
  
  logrho ~ normal(-2, 0.5);
  sigma_obs ~ normal(0, 0.01);
  S0 ~ normal(0.3, 0.3);
  logx_I0 ~ normal(-4, 2);
  betaphase ~ uniform(0, 2 * pi());
  dbeta ~ normal(0, 0.10);

  // Likelihood: use the observed (quarterly) data.
  for (i in 1:N) {
    if (positivity[i] > 0) {
      positivity[i] ~ normal(Ifit[i], sigma_obs);
    }
  }
}

generated quantities {
    print("I0=", I0, " S0=", S0, " dbeta=", dbeta, " betaphase=", betaphase, " sigma_obs=", sigma_obs, " rho=", rho, " beta0=", beta0);
}
