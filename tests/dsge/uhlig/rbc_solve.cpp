/*################################################################################
  ##
  ##   Copyright (C) 2011-2018 Keith O'Hara
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
 * Solving a basic RBC model using Uhlig's method
 */

#include "bmlib.hpp"

int main(int argc, char** argv)
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

    double C52 = -(alpha/RSS)*YKSS;
    
    arma::mat A, B, C, D, F, G, H, J, K, L, M, N;
    
    A.zeros(5,1); A(2,0) = 1;
    
    B.zeros(5,1);
    B(1,0) = - alpha; B(2,0) = -(1-delta); B(4,0) = -C52;

    //              c          y           l          r          i
    C      <<   -CYSS <<       1 <<        0 <<       0 <<   -IYSS << arma::endr
           <<       0 <<       1 << -1+alpha <<       0 <<       0 << arma::endr
           <<       0 <<       0 <<        0 <<       0 <<   -IKSS << arma::endr
           <<     eta <<      -1 <<        1 <<       0 <<       0 << arma::endr
           <<       0 <<     C52 <<        0 <<       1 <<       0 << arma::endr;
    
    D.zeros(5,1); 
    D(1,0) = -1.0;
    
    F.zeros(1,1); G.zeros(1,1); H.zeros(1,1);
    
    J.zeros(1,5); 
    J(0,0) = -1; 
    J(0,3) = 1/eta;
    
    K.zeros(1,5); 
    K(0,0) = 1;
    
    L.zeros(1,1); M.zeros(1,1);
    
    N.zeros(1,1); 
    N(0,0) = rho;

    //

    arma::mat P, Q, R, S;
    arma::cx_vec eigenvals; arma::cx_mat eigenvecs;
    
    //arma::vec whicheig(2); whicheig(0) = 2; whicheig(1) = 1;
    
    bm::uhlig_solver(A, B, C, D, F, G, H, J, K, L, M, N, P, Q, R, S);
    bm::uhlig_solver(A, B, C, D, F, G, H, J, K, L, M, N, nullptr, P, Q, R, S, &eigenvals, &eigenvecs);

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
//
//
//END