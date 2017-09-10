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
 * Solving a basic NKM model
 */

#include "bmlib.hpp"

int main()
{
    //
    double alpha    = 0.33;
    double beta     = 0.99;
    double vartheta = 6.0;
    double theta    = 0.6667;
    
    double eta      = 1.0;
    double phi      = 1.0;
    double phi_pi   = 1.5;
    double phi_y    = 0.125;
    double rho_a    = 0.90;
    double rho_v    = 0.5;

    double BigTheta = (1-alpha)/(1-alpha+alpha*vartheta);
    double kappa = (((1-theta)*(1-beta*theta))/(theta))*BigTheta*((1/eta)+((phi+alpha)/(1-alpha)));
    double psi = (eta*(1+phi))/(1-alpha+eta*(phi + alpha));
    
    //

    double G0_47 = (1/eta)*psi*(rho_a - 1);
    
    arma::mat Gamma0(10,10);
    arma::mat Gamma1(10,10);

    //            y_g           y          pi         r_n           i           n           a           v
    Gamma0 <<      -1  <<       0  <<       0  <<     eta  <<  -eta/4  <<       0  <<       0  <<       0  <<       1  <<   eta/4  << arma::endr
           <<   kappa  <<       0  <<   -0.25  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<  beta/4  << arma::endr
           <<   phi_y  <<       0  <<phi_pi/4  <<       0  <<   -0.25  <<       0  <<       0  <<       1  <<       0  <<       0  << arma::endr
           <<       0  <<       0  <<       0  <<      -1  <<       0  <<       0  <<   G0_47  <<       0  <<       0  <<       0  << arma::endr
           <<       0  <<      -1  <<       0  <<       0  <<       0  << 1-alpha  <<       1  <<       0  <<       0  <<       0  << arma::endr
           <<      -1  <<       1  <<       0  <<       0  <<       0  <<       0  <<    -psi  <<       0  <<       0  <<       0  << arma::endr
           <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       1  <<       0  <<       0  <<       0  << arma::endr
           <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       1  <<       0  <<       0  << arma::endr
           <<       1  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  << arma::endr
           <<       0  <<       0  <<       1  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  << arma::endr;
    
    Gamma1 <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  << arma::endr
           <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  << arma::endr
           <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  << arma::endr
           <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  << arma::endr
           <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  << arma::endr
           <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  << arma::endr
           <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<   rho_a  <<       0  <<       0  <<       0  << arma::endr
           <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<   rho_v  <<       0  <<       0  << arma::endr
           <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       1  <<       0  << arma::endr
           <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       1  << arma::endr;
    
    //
    
    arma::mat C = arma::zeros<arma::mat>(10,1);
    
    arma::mat Psi = arma::zeros<arma::mat>(10,2);
    Psi(6,0) = 1;
    Psi(7,1) = 1;
    
    arma::mat Pi = arma::zeros<arma::mat>(10,2);
    Pi(8,0) = 1;
    Pi(9,1) = 1;

    //
    // setup solution matrices

    arma::mat G1; arma::mat Cons; arma::mat impact;
    arma::cx_mat fmat; arma::cx_mat fwt; arma::cx_mat ywt; arma::cx_mat gev;
    arma::vec eu; arma::mat loose;
    
    bm::gensys_solver(Gamma0, Gamma1, C, Psi, Pi, G1, Cons, impact, &fmat, &fwt, &ywt, &gev, &eu, &loose);
    bm::gensys_solver(Gamma0, Gamma1, C, Psi, Pi, G1, Cons, impact);
    
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

    arma::mat shocks_cov = arma::eye(2,2);
    shocks_cov(1,1) = 0.25*0.25;
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
}
