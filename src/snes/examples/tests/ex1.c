#ifndef lint
static char vcid[] = "$Id: dsfi.c,v 1.2 1994/09/01 21:18:20 curfman Exp $";
#endif

#include "user.h"
#include <math.h>

int Dsfi( nlP, x, fvec, iflag )
NLCtx  *nlP;
double *x, *fvec;
int    iflag;
/*  
    This routine generates an initial guess, a function, and a Jacobian 
    matrix on a single processor for the Solid Fuel Ignition (SFI) problem 
    from the MINPACK-2 suite.  This problem is modeled by the partial 
    differential equation
  
            -Laplacian u - lambda*exp(u) = 0,  0 < x,y < 1 ,
  
    with boundary conditions
   
             u = 0  for  x = 0, x = 1, y = 0, y = 1.
  
    A finite difference approximation with the usual 5-point stencil
    is used to discretize the boundary value problem to obtain a nonlinear 
    system of equations.
  
    The nonzero terms of the Jacobian are stored in a sparse format with 
    the SpMat data structure and are added to the Jacobian with 
    "SpAddValue" calls. 
 */
{
#define MIN(a,b) ( ((a)<(b)) ? a : b )

  UserCtx *user;
  SpMat   *jac;
  int     i, j, row, mx, my;
  double  two = 2.0, one = 1.0, zero = 0.0, lambda;
  double  temp1, temp, hx, hy, hxdhy, hydhx;
  double  ut, ub, ul, ur, u, uxx, uyy, sc;
  double  xnorm, bratu_lambda_max = 6.81, bratu_lambda_min = 0.;

  jac	 = (SpMat *)NLGetMatrixCtx( nlP );
  user	 = (UserCtx *)NLGetUserCtx( nlP );
  mx	 = user->mx; 
  my	 = user->my;
  lambda = user->problem.param;

  /* Test for invalid input parameters */
  if ( (mx <= 0) || (my <= 0) ) return 1;
  if ( (lambda > bratu_lambda_max) || (lambda <= bratu_lambda_min) ) return 2; 

  hx    = one / (double)(mx-1);
  hy    = one / (double)(my-1);
  sc    = hx*hy;
  hxdhy = hx/hy;
  hydhx = hy/hx;

  switch (iflag) {

     /* Compute Initial Guess */
     case -1: 
        temp1 = lambda/(lambda + one);
        for (j=0; j<my; j++) {
           temp = (double)(MIN(j,my-j-1))*hy;
           for (i=0; i<mx; i++) {
              row = i + j*mx;  
              if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
                  x[row] = 0.0; 
                  continue;
                  }
               x[row] = temp1*sqrt( MIN( (double)(MIN(i,mx-i-1))*hx,temp) ); 
              }
           }
        return 0; 

     /* Evaluate Function */
     case 1: 
        for (j=0; j<my; j++) {
           for (i=0; i<mx; i++) {
              row = i + j*mx;
              if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
                  fvec[row] = 0.0;
                  continue;
                  }
              u = x[row];
              ub = x[row - mx];
              ul = x[row - 1];
              ut = x[row + mx];
              ur = x[row + 1];
              uxx = (-ur + two*u - ul)*hydhx;
              uyy = (-ut + two*u - ub)*hxdhy;
              fvec[row] = uxx + uyy - sc*lambda*exp(x[row]);
              }
           }
        return 0; 

     /* Evaluate Jacobian matrix */
     case 2:

        for (j=0; j<my; j++) {
           for (i=0; i<mx; i++) {
              row = i + j*mx;
              if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
                  SpAddValue( jac, 1.0, row, row );
                  continue;
                  }
              SpAddValue( jac, -hxdhy, row, row - mx );
              SpAddValue( jac, -hydhx, row, row - 1 );
              SpAddValue( jac,
                 two*(hydhx + hxdhy) - sc*lambda*exp(x[row]) , row, row );
              SpAddValue( jac, -hydhx, row, row + 1 );
              SpAddValue( jac, -hxdhy, row, row + mx );
              }
          }
     return 0;
  }
}
