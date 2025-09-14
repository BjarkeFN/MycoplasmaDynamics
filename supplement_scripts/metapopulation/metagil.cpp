#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>

struct Update {
    int idx;
    int dS;
    int dI;
    int dR;
};

int main() {
    // Parameters (hard-coded)
    double beta0  = 0.649645859;
    double gamma  = 1.0/2.5;
    double mu     = 1.0/(80*52);
    double delta  = 0.00230892;
    double dbeta  = 0.18228653;
    double c_within = 0.99999;
    double T_max = 150*52; // weeks
    int random_seed = 84; // for reproducibility

    int M = 5; // number of subpopulations
    int N0 = 500000; // (initial) size of each subpop
    std::vector<int> S(M, int(0.6458905435*N0)); // initial susceptible
    std::vector<int> I(M, int(0.0014751413*N0)); // initial infected
    std::vector<int> R(M, N0);
    for (int i=0; i<M; ++i) {
        R[i] -= (S[i] + I[i]); // since R was initiated at N0
    }

    double t = 0.0;

    // Output
    std::ofstream fout("sirs_metapop.csv");
    fout << "t";
    for (int i=0; i<M; ++i) fout << ",S" << i;
    for (int i=0; i<M; ++i) fout << ",I" << i;
    for (int i=0; i<M; ++i) fout << ",R" << i;
    fout << "\n";

    auto write_state = [&](double t) {
        fout << t;
        for (int i=0; i<M; ++i) fout << "," << S[i];
        for (int i=0; i<M; ++i) fout << "," << I[i];
        for (int i=0; i<M; ++i) fout << "," << R[i];
        fout << "\n";
    };

    write_state(t);

    std::mt19937 rng(random_seed);
    std::uniform_real_distribution<double> U(0.0,1.0);

    double last_record_t = 0.0;

    while (t < T_max) {
        // compute current N
        std::vector<int> N(M);
        int totalN = 0;
        for (int i=0; i<M; ++i) {
            N[i] = S[i] + I[i] + R[i];
            totalN += N[i];
        }
        if (totalN <= 0) break;

        double beta_t = beta0 * (1.0 + dbeta * std::sin(2*M_PI*t/52.0));

        std::vector<double> rates;
        std::vector<Update> updates;

        for (int i=0; i<M; ++i) {
            int Ni = N[i];

            rates.push_back(mu*Ni);
            updates.push_back({i, +1, 0, 0}); // births

            rates.push_back(mu*S[i]);
            updates.push_back({i, -1, 0, 0}); // deaths in the S state
            
            rates.push_back(mu*I[i]);
            updates.push_back({i, 0, -1, 0}); // deaths in the I state
            
            rates.push_back(mu*R[i]);
            updates.push_back({i, 0, 0, -1}); // Deaths in the R state
            
            rates.push_back(beta_t * c_within * S[i] * I[i] / double(Ni));
            updates.push_back({i, -1, +1, 0}); // infections within-subpop

            for (int j=0; j<M; ++j) {
                rates.push_back(beta_t * ((1.0-c_within)/(double(M)-1)) * S[i] * I[j] / double(N[j]));
                updates.push_back({i, -1, +1, 0}); // cross-subpop infections
            }

            rates.push_back(gamma*I[i]); 
            updates.push_back({i, 0, -1, +1}); // recovery

            rates.push_back(delta*R[i]);
            updates.push_back({i, +1, 0, -1}); // waning
        }

        double total_rate = 0.0;
        for (auto r : rates) total_rate += r;
        if (total_rate <= 0) break;

        // time step
        double r1 = U(rng);
        double dt = -std::log(r1)/total_rate;
        t += dt;
        if (t > T_max) break;

        // pick event
        double r2 = U(rng)*total_rate;
        double cumulative = 0.0;
        int chosen = -1;
        for (size_t k=0; k<rates.size(); ++k) {
            cumulative += rates[k];
            if (r2 <= cumulative) {
                chosen = k;
                break;
            }
        }
        
        auto u = updates[chosen];
        S[u.idx] += u.dS;
        I[u.idx] += u.dI;
        R[u.idx] += u.dR;
        
        if (I[u.idx] < 1) {
         I[u.idx] = 1; // prevent from dying out completely
         }

        // thinning output
        if (t - last_record_t > 1.0/7.0) { // if more than a day has passed
            write_state(t);
            if (int(t/52) > int(last_record_t/52)) {
                std::cout << "t (years) = " << int(t/52) << "\n";
            }
            last_record_t = t;
        }
    }

    fout.close();
    return 0;
}

