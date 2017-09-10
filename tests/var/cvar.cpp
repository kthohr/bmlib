/*################################################################################
  ##
  ##   Copyright (C) 2011-2017 Keith O'Hara
  ##
  ##   This file is part of the BMLib C++ library.
  ##
  ##   BMLib is free software: you can redistribute it and/or modify
  ##   it under the terms of the GNU General Public License as published by
  ##   the Free Software Foundation, either version 2 of the License, or
  ##   (at your option) any later version.
  ##
  ##   BMLib is distributed in the hope that it will be useful,
  ##   but WITHOUT ANY WARRANTY; without even the implied warranty of
  ##   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  ##   GNU General Public License for more details.
  ##
  ##   You should have received a copy of the GNU General Public License
  ##   along with BMLib. If not, see <http://www.gnu.org/licenses/>.
  ##
  ################################################################################*/

/*
 * cvar example
 */

#include "bmlib.hpp"

int main()
{
    bm::comptime begin_time = bm::tic();

    int n = 1000;
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
    
    bm::cvar cvar_model;

    cvar_model.build(Y,cons_term,p);

    //
    
    cvar_model.estim();

    arma::cout << "beta_hat:\n" << cvar_model.beta_hat << arma::endl;
    arma::cout << "Sigma_hat:\n" << cvar_model.Sigma_hat << arma::endl;
    
    //

    int n_draws  = 10000;

    cvar_model.boot(n_draws);

    //

    int n_irf_periods = 20;

    cvar_model.IRF(n_irf_periods);

    arma::cout << "IRFs:\n" << cvar_model.irfs.slice(0) << arma::endl;

    //

    int forecast_horizon = 10;

    arma::cube forecasts = cvar_model.forecast(forecast_horizon,false);

    arma::cout << "Forecasts:\n" << forecasts.slice(0) << arma::endl;
    
    //

    bm::tictoc(begin_time);

    return 0;
}
