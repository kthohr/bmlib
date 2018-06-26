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
 * Solving a basic RBC model using gensys++
 */

#include "bmpp.hpp"

int main()
{
    //

    double beta    = 0.99;
    double alpha   = .33;
    double delta   = .015;
    double eta     = 1.0;
    double rho     = 0.95;
    double sigma_T = 1.0;
    
    double RSS  = 1/beta;
    double YKSS = (RSS + delta - 1)/alpha;
    double IKSS = delta;
    double IYSS = ((alpha*delta)/(RSS + delta - 1));
    double CYSS = 1 - IYSS;
    
    //

    double Gam62 = alpha*YKSS/(RSS);
    double Gam63 = alpha*YKSS/(RSS);
    
    arma::mat Gamma0(9,9);
    arma::mat Gamma1(9,9);
    //            c_t,       y_t,       k_t,       n_t,       r_t,       i_t,       a_t,     c_t+1,     r_t+1
    Gamma0 <<       1 <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 <<      -1 <<   1/eta << arma::endr
           <<   -CYSS <<       1 <<       0 <<       0 <<       0 <<   -IYSS <<       0 <<       0 <<       0 << arma::endr
           <<       0 <<       1 <<       0 <<-1+alpha <<       0 <<       0 <<      -1 <<       0 <<       0 << arma::endr
           <<       0 <<       0 <<       1 <<       0 <<       0 <<   -IKSS <<       0 <<       0 <<       0 << arma::endr
           <<    -eta <<       1 <<       0 <<      -1 <<       0 <<       0 <<       0 <<       0 <<       0 << arma::endr
           <<       0 <<   Gam62 <<       0 <<       0 <<      -1 <<       0 <<       0 <<       0 <<       0 << arma::endr
           <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 <<       1 <<       0 <<       0 << arma::endr
           <<       1 <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 << arma::endr
           <<       0 <<       0 <<       0 <<       0 <<       1 <<       0 <<       0 <<       0 <<       0 << arma::endr;
    
    Gamma1 <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 << arma::endr
           <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 << arma::endr
           <<       0 <<       0 <<   alpha <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 << arma::endr
           <<       0 <<       0 << 1-delta <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 << arma::endr
           <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 << arma::endr
           <<       0 <<       0 <<   Gam63 <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 << arma::endr
           <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 <<     rho <<       0 <<       0 << arma::endr
           <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 <<       1 <<       0 << arma::endr
           <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 <<       0 <<       1 << arma::endr;
    
    //

    arma::mat C = arma::zeros<arma::mat>(9,1);
    
    arma::mat Psi = arma::zeros<arma::mat>(9,1);
    Psi(6,0) = 1;
    
    arma::mat Pi = arma::zeros<arma::mat>(9,2);
    Pi(7,0) = 1; Pi(8,1) = 1;
    
    //

    arma::mat G1; arma::mat Cons; arma::mat impact;
    arma::cx_mat fmat; arma::cx_mat fwt; arma::cx_mat ywt; arma::cx_mat gev;
    arma::vec eu; arma::mat loose;
    
    bm::gensys_solver(Gamma0, Gamma1, C, Psi, Pi, G1, Cons, impact, &fmat, &fwt, &ywt, &gev, &eu, &loose);

    //
    
    arma::cout << "G1:\n" << G1 << arma::endl;
    arma::cout << "Cons:\n" << Cons << arma::endl;
    arma::cout << "impact:\n" << impact << arma::endl;

    arma::cout << "eu:\n" << eu << arma::endl;
    arma::cout << "fmat:\n" << fmat << arma::endl;
    arma::cout << "fwt:\n" << fwt << arma::endl;
    arma::cout << "ywt:\n" << ywt << arma::endl;
    arma::cout << "gev:\n" << gev << arma::endl;
    arma::cout << "loose:\n" << loose << arma::endl;

    //
    // build a gensys model

    bm::gensys model_obj;

    model_obj.build(Gamma0,Gamma1,C,Psi,Pi);

    model_obj.solve();

    std::cout << "solution from class input\n" << std::endl;

    arma::cout << "G1:\n" << model_obj.G_sol << arma::endl;
    arma::cout << "cons:\n" << model_obj.cons_sol << arma::endl;
    arma::cout << "impact:\n" << model_obj.impact_sol << arma::endl;

    //
    // simulate from the model

    arma::mat shocks_cov = arma::eye(1,1);
    shocks_cov(0,0) = sigma_T*sigma_T;
    model_obj.shocks_cov = shocks_cov;

    arma::mat model_data = model_obj.simulate(10000,1000);
    arma::cout << "data cov:\n" << arma::cov(model_data) << arma::endl;

    model_obj.state_space();

    // steady-state covariance matrix

    arma::mat GQG = model_obj.G_state*shocks_cov*model_obj.G_state.t();
    arma::mat ss_cov = bm::lyapunov_dbl(GQG,model_obj.F_state); // steady-state covariance matrix

    arma::cout << "model cov:\n" << ss_cov << arma::endl;

    //

    return 0;

    //

    return 0;
}
