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
 * Estimate a basic New Keynesian model
 */

#include "bmpp.hpp"
#include "nkm_model.hpp"

int main()
{
    bm::comptime begin_time = bm::tic();

    //
    // setup

    bm::dsge<bm::gensys> dsge_obj;

    dsge_obj.model_fn = nkm_model_simple_fn;

    arma::vec pars_init = arma::ones(1,1);

    dsge_obj.solve_to_state_space(pars_init);

    arma::mat model_data = dsge_obj.lrem_obj.simulate(200,1000);
    model_data = arma::join_rows(model_data.col(2),model_data.col(4));

    dsge_obj.estim_data = model_data;

    // prior and bounds

    const int n_pars = 1;

    dsge_obj.prior_form.set_size(n_pars);
    dsge_obj.prior_form(0) = 1;

    dsge_obj.prior_pars.set_size(n_pars,2);
    dsge_obj.prior_pars(0,0) = 1.0;
    dsge_obj.prior_pars(0,1) = 0.05;

    dsge_obj.lower_bounds.set_size(n_pars,1);
    dsge_obj.upper_bounds.set_size(n_pars,1);

    dsge_obj.lower_bounds(0) = 0.5;
    dsge_obj.upper_bounds(0) = 10.0;

    arma::vec initial_vals(n_pars);
    initial_vals(0) = 1.0;

    // mode estim and mcmc

    optim::algo_settings o_settings;

    o_settings.de_initial_lb = dsge_obj.lower_bounds;
    o_settings.de_initial_ub = dsge_obj.upper_bounds;

    o_settings.de_check_freq = 50;

    arma::mat vcov_mat;
    arma::vec res = dsge_obj.estim_mode(initial_vals,&vcov_mat,&o_settings);
    arma::cout << "mode:\n" << res << arma::endl;
    arma::cout << "var-cov matrix:\n" << vcov_mat << arma::endl;

    arma::cube check_vals = bm::mode_check(dsge_obj,res,11);
    arma::cout << "check_vals:\n" << check_vals << arma::endl;

    // try Chandrasekhar recursions

    dsge_obj.filter_choice = 2;

    res = dsge_obj.estim_mode(initial_vals,&vcov_mat,&o_settings);
    arma::cout << "mode (Chand):\n" << res << arma::endl;

    dsge_obj.filter_choice = 1; // back to Kalman

    //

    mcmc::algo_settings_t m_settings;

    m_settings.de_initial_lb = dsge_obj.lower_bounds;
    m_settings.de_initial_ub = dsge_obj.upper_bounds;

    m_settings.de_n_burnin = 100;
    m_settings.de_n_gen = 100;

    dsge_obj.estim_mcmc(initial_vals,&m_settings);

    arma::cout << "mcmc mean:\n" << arma::mean(dsge_obj.dsge_draws) << arma::endl;

    arma::cube fcast = dsge_obj.forecast(10,true);

    //

    bm::tictoc(begin_time);

    return 0;
}
