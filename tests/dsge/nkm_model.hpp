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

// Basic New-Keynesian Model

void
nkm_model_fn(const arma::vec& pars_inp, bm::gensys& model_obj, arma::mat& shocks_cov_out, arma::mat& C_out, arma::mat& H_out, arma::mat& R_out)
{
    double alpha    = pars_inp(0);
    double beta     = pars_inp(1);
    double vartheta = pars_inp(2);
    double theta    = pars_inp(3);
    
    double eta    = pars_inp(4);
    double phi    = pars_inp(5);
    double phi_pi = pars_inp(6);
    double phi_y  = pars_inp(7);

    double rho_a  = pars_inp(8);
    double rho_v  = pars_inp(9);

    double sigma_T = pars_inp(10);
    double sigma_M = pars_inp(11);

    //

    double BigTheta = (1-alpha)/(1-alpha+alpha*vartheta);
    double kappa = (((1-theta)*(1-beta*theta))/(theta)) * BigTheta * ((1.0/eta)+((phi+alpha)/(1-alpha)));
    double psi = (eta*(1+phi))/(1-alpha+eta*(phi + alpha));

    //

    model_obj.Gamma_0.set_size(10,10);
    model_obj.Gamma_1.set_size(10,10);

    double G0_47 = (1.0/eta)*psi*(rho_a - 1);

    model_obj.Gamma_0 <<
                 -1.0  <<       0  <<          0  <<     eta  << -eta/4.0  <<           0  <<       0  <<       0  <<     1.0  <<   eta/4.0  << arma::endr
           <<   kappa  <<       0  <<      -0.25  <<       0  <<        0  <<           0  <<       0  <<       0  <<       0  <<  beta/4.0  << arma::endr
           <<   phi_y  <<       0  << phi_pi/4.0  <<       0  <<    -0.25  <<           0  <<       0  <<     1.0  <<       0  <<         0  << arma::endr
           <<       0  <<       0  <<          0  <<    -1.0  <<        0  <<           0  <<   G0_47  <<       0  <<       0  <<         0  << arma::endr
           <<       0  <<    -1.0  <<          0  <<       0  <<        0  << 1.0 - alpha  <<     1.0  <<       0  <<       0  <<         0  << arma::endr
           <<    -1.0  <<     1.0  <<          0  <<       0  <<        0  <<           0  <<    -psi  <<       0  <<       0  <<         0  << arma::endr
           <<       0  <<       0  <<          0  <<       0  <<        0  <<           0  <<     1.0  <<       0  <<       0  <<         0  << arma::endr
           <<       0  <<       0  <<          0  <<       0  <<        0  <<           0  <<       0  <<     1.0  <<       0  <<         0  << arma::endr
           <<     1.0  <<       0  <<          0  <<       0  <<        0  <<           0  <<       0  <<       0  <<       0  <<         0  << arma::endr
           <<       0  <<       0  <<        1.0  <<       0  <<        0  <<           0  <<       0  <<       0  <<       0  <<         0  << arma::endr;

    model_obj.Gamma_1 <<
                    0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  << arma::endr
           <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  << arma::endr
           <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  << arma::endr
           <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  << arma::endr
           <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  << arma::endr
           <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  << arma::endr
           <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<   rho_a  <<       0  <<       0  <<       0  << arma::endr
           <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<   rho_v  <<       0  <<       0  << arma::endr
           <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<     1.0  <<       0  << arma::endr
           <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<       0  <<     1.0  << arma::endr;

    //

    model_obj.Gamma_C = arma::zeros<arma::mat>(10,1);

    model_obj.Psi = arma::zeros<arma::mat>(10,2);
    model_obj.Psi(6,0) = 1;
    model_obj.Psi(7,1) = 1;

    model_obj.Pi = arma::zeros<arma::mat>(10,2);
    model_obj.Pi(8,0) = 1;
    model_obj.Pi(9,1) = 1;

    //

    shocks_cov_out.zeros(2,2);
    shocks_cov_out(0,0) = sigma_T*sigma_T;
    shocks_cov_out(1,1) = sigma_M*sigma_M;

    C_out.zeros(2,1);

    H_out.zeros(8,2);
    H_out(2,0) = 1.0;
    H_out(4,1) = 1.0;

    R_out.zeros(2,2);
}

void
nkm_model_simple_fn(const arma::vec& pars_inp, bm::gensys& model_obj, arma::mat& shocks_cov_out, arma::mat& C_out, arma::mat& H_out, arma::mat& R_out)
{

    double eta    = pars_inp(0);

    //

    arma::vec pars_new(12);

    pars_new(0)  = 0.33;   // alpha
    pars_new(1)  = 0.99;   // beta
    pars_new(2)  = 6.0;    // vartheta
    pars_new(3)  = 0.6667; // theta

    pars_new(4)  = eta;
    pars_new(5)  = 1.0;    // phi
    pars_new(6)  = 1.5;    // phi_pi
    pars_new(7)  = 0.125;  // phi_y

    pars_new(8)  = 0.9;    // rho_a
    pars_new(9)  = 0.5;    // rho_v

    pars_new(10) = 1.0;    // sigma_T
    pars_new(11) = 0.25;   // sigma_M
    
    //

    nkm_model_fn(pars_new,model_obj,shocks_cov_out,C_out,H_out,R_out);

}
