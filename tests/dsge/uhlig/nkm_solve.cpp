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
 * Solving a basic NKM model using Uhlig's method
 */

#include "bmpp.hpp"

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

    double sigma_T  = 1.0;
    double sigma_M  = 0.25;
    
    double BigTheta = (1-alpha)/(1-alpha+alpha*vartheta);
    double kappa    = (((1-theta)*(1-beta*theta))/(theta))*BigTheta*((1/eta)+((phi+alpha)/(1-alpha)));
    double psi      = (eta*(1+phi))/(1-alpha+eta*(phi + alpha));

    //

    double G33 = -phi_pi/4;
    double M41 = -(1/eta)*psi*(rho_a - 1);
    
    arma::mat A, B, C, D, F, G, H, J, K, L, M, N;
    
    //

    F.zeros(6,6); G.zeros(6,6); H.zeros(6,6);
    F(0,0) = -1.0; F(0,2) = -eta/4.0; F(1,2) = -beta/4.0;

    //             yg          y          pi         rn          i          n
    G      <<       1 <<       0 <<        0 <<    -eta <<   eta/4.0 <<             0 << arma::endr
           <<  -kappa <<       0 <<     0.25 <<       0 <<         0 <<             0 << arma::endr
           <<  -phi_y <<       0 <<      G33 <<       0 <<      0.25 <<             0 << arma::endr
           <<       0 <<       0 <<        0 <<       1 <<         0 <<             0 << arma::endr
           <<       0 <<       1 <<        0 <<       0 <<         0 <<  -1.0 + alpha << arma::endr
           <<       1 <<      -1 <<        0 <<       0 <<         0 <<             0 << arma::endr;
    
    L.zeros(6,2);
    
    M.zeros(6,2); 
    M(2,1) = -1.0; 
    M(3,0) = M41; 
    M(4,0) = -1.0; 
    M(5,0) = psi;
    
    N.zeros(2,2); 
    N(0,0) = rho_a; 
    N(1,1) = rho_v;

    //

    arma::mat P, Q, R, S;
    arma::cx_vec eigenvals; arma::cx_mat eigenvecs;
    
    // arma::vec whicheig(2); whicheig(0) = 2; whicheig(1) = 1;
    
    bm::uhlig_solver(A, B, C, D, F, G, H, J, K, L, M, N, nullptr, P, Q, R, S, &eigenvals, &eigenvecs);
    bm::uhlig_solver(A, B, C, D, F, G, H, J, K, L, M, N, P, Q, R, S);
    
    //

    arma::cout << "P sol:\n" << P << arma::endl;
    arma::cout << "Q sol:\n" << Q << arma::endl;
    arma::cout << "R sol:\n" << R << arma::endl;
    arma::cout << "S sol:\n" << S << arma::endl;

    arma::cout << "eigen values:\n" << eigenvals << arma::endl;
    arma::cout << "eigen vectors:\n" << eigenvecs << arma::endl;

    //

    return 0;
}
