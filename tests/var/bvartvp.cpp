/*
 * BVAR-TVP example
 *
 * cd ~/Desktop/SCM/Bitbucket/BMLib/Examples/bvar
 * g++-mp-7 -std=c++11 -O2 -Wall -I/opt/local/include -I./../../include bvartvp.cpp -o bvartvp.test -framework Accelerate -L./../.. -lbm
 */

#include "bmlib.hpp"

int main()
{
    bm::comptime begin_time = bm::tic();

    int n = 240;
    int sim_burn = 1000;

    bool cons_term = true;
    int p = 2;
    // int M = 2;

    arma::mat beta(5,2);
    beta <<  7.0  <<  3.0  << arma::endr
         <<  0.50 <<  0.20 << arma::endr
         <<  0.28 <<  0.70 << arma::endr
         << -0.39 << -0.10 << arma::endr
         <<  0.10 <<  0.05 << arma::endr;

    //

    arma::mat Y = bm::var_sim(beta,cons_term,n,sim_burn);

    //

    bm::bvartvp bvar_model;

    bvar_model.build(Y,cons_term,p);

    //

    int tau = 40;

    double Xi_beta = 4;
    double Xi_Q = 0.0001;
    double Xi_Sigma = 1.0;

    double gamma_S = 4;
    double gamma_Q = tau;

    bvar_model.prior(tau,Xi_beta,Xi_Q,gamma_Q,Xi_Sigma,gamma_S);

    //

    int n_burnin = 5000;
    int n_draws  = 5000;

    bvar_model.gibbs(n_draws,n_burnin);

    arma::cout << "alpha mean:\n" << bvar_model.alpha_pt_mean.t() << arma::endl;

    //

    bm::tictoc(begin_time);

    return 0;
}
