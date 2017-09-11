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
 * Estimate a DSGE-VAR
 */

#include "bmlib.hpp"
#include "nkm_model.hpp"

int main()
{
    bm::comptime begin_time = bm::tic();
    
    //
    // setup

    bm::dsgevar<bm::gensys> dsgevar_obj;

    dsgevar_obj.dsge_obj.model_fn = nkm_model_simple_fn;

    arma::vec pars_init = arma::ones(1,1);

    dsgevar_obj.dsge_obj.solve_to_state_space(pars_init);

    arma::mat model_data = dsgevar_obj.dsge_obj.lrem_obj.simulate(200,1000);
    model_data = arma::join_rows(model_data.col(2),model_data.col(4));

    //

    const double lambda = 1.0;

    dsgevar_obj.build(model_data,false,1,lambda);

    // prior and bounds

    const int n_pars = 1;

    dsgevar_obj.dsge_obj.prior_form.set_size(n_pars);
    dsgevar_obj.dsge_obj.prior_form(0) = 1;

    dsgevar_obj.dsge_obj.prior_pars.set_size(n_pars,2);
    dsgevar_obj.dsge_obj.prior_pars(0,0) = 1.0;
    dsgevar_obj.dsge_obj.prior_pars(0,1) = 0.05;

    dsgevar_obj.dsge_obj.lower_bounds.set_size(n_pars,1);
    dsgevar_obj.dsge_obj.upper_bounds.set_size(n_pars,1);

    dsgevar_obj.dsge_obj.lower_bounds(0) = 0.5;
    dsgevar_obj.dsge_obj.upper_bounds(0) = 5.0;

    arma::vec initial_vals(n_pars);
    initial_vals(0) = 1.0;

    //
    // mode estim and mcmc

    optim::opt_settings o_settings;

    o_settings.de_initial_lb = dsgevar_obj.dsge_obj.lower_bounds;
    o_settings.de_initial_ub = dsgevar_obj.dsge_obj.upper_bounds;

    o_settings.de_check_freq = 50;

    arma::vec res = dsgevar_obj.estim_mode(initial_vals,&o_settings);
    arma::cout << "mode:\n" << res << arma::endl;

    //

    mcmc::mcmc_settings m_settings;

    m_settings.de_initial_lb = dsgevar_obj.dsge_obj.lower_bounds;
    m_settings.de_initial_ub = dsgevar_obj.dsge_obj.upper_bounds;

    m_settings.de_n_burnin = 100;
    m_settings.de_n_gen = 100;

    dsgevar_obj.estim_mcmc(initial_vals,&m_settings);

    arma::cout << "mcmc mean:\n" << arma::mean(dsgevar_obj.dsge_obj.dsge_draws) << arma::endl;

    //

    bm::tictoc(begin_time);

    return 0;
}
