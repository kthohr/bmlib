/*################################################################################
  ##
  ##   Copyright (C) 2011-2018 Keith O'Hara
  ##
  ##   This file is part of the BM++ C++ library.
  ##
  ##   BM++ is free software: you can redistribute it and/or modify
  ##   it under the terms of the GNU General Public License as published by
  ##   the Free Software Foundation, either version 2 of the License, or
  ##   (at your option) any later version.
  ##
  ##   BM++ is distributed in the hope that it will be useful,
  ##   but WITHOUT ANY WARRANTY; without even the implied warranty of
  ##   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  ##   GNU General Public License for more details.
  ##
  ##   You should have received a copy of the GNU General Public License
  ##   along with BM++. If not, see <http://www.gnu.org/licenses/>.
  ##
  ################################################################################*/

/*
 * bvartvp example
 */

#include "bmpp.hpp"

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
