#include <math.h>

/*------------------------------------------------------*/
/* forcing fucntions */
 double pde_g(double x,double y){ 
  return 0; } 
 double pde_f(double x,double y){
   return 0; }
/* Boundary Conditions */
/* use the  exact solution for the burgers equation */

/*-------------------------------------------------
double pde_bc1(double x,double y)
{
  double eta = 0.04;
  double a1,a2,a3,a4;
  a1 = 30,a2 = 1, a3 = 5,a4 = 20;
  return -2*eta*(a2+a4*y)/(a1+a2*x+a3*y+a4*x*y);
}
double pde_bc2(double x,double y)
{
   double eta = 0.04;
   double a1,a2,a3,a4;
   a1 = 30,a2 = 1,a3 = 5,a4 = 20;
   return -2*eta*(a3+a4*x)/(a1+a2*x+a3*y+a4*x*y);
}
double pde_soln(double x,double y)
{
  double a1,a2,a3,a4;
  a1 = 30,a2 = 1,a3 = 5,a4 = 20;
  return a1+a2*x+a3*y+a4*x*y;
}
*/
/*--------------------------------------------------------*/

double pde_soln(double x,double y)
{
  double a1,a2,a3,a4,a5,lambda;
  a1 = 1.3e13;a2 = 1.3e13;a3 = 0;a4 = 0;a5 = 1;
  lambda = 25;
  return a1+a2*(x+1) +a3*y+a4*(x+1)*y+a5*(exp(lambda*x)+exp(-lambda*x))*cos(lambda*y);
}

/*------------------------------------------------------------*/
double pde_bc1(double x,double y)
  {
    double a2,a4,a5,lambda,eta;
    double q;
    a2 = 1.3e13;a4 = 0;a5 = 1;
    lambda = 25; eta = 0.04;
    q = pde_soln(x,y);
  return -2*eta*(a2+a4*y+lambda*a5*(exp(lambda*x)-exp(-lambda*x))*cos(lambda*y))/q;
}
/*----------------------------------------------------------*/

double pde_bc2(double x,double y)
{
  double a3,a4,a5,lambda,eta;
  double q;
  a3 = 0;a4 = 0;a5 = 1;
  lambda = 25; eta = 0.04;
   q = pde_soln(x,y);
   return -2*eta*(a3+a4*(x+1)-lambda*a5*(exp(lambda*x)+exp(-lambda*x))*sin(lambda*y))/q;
}





