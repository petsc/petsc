
#include <math.h>

/* forcing fucntions */
 PetscReal pde_g(PetscReal x,PetscReal y){ 
  return 0; } 
 PetscReal pde_f(PetscReal x,PetscReal y){
   return 0; }
/* Boundary Conditions */
/* use the  exact solution for the burgers equation */

/*
PetscReal pde_bc1(PetscReal x,PetscReal y)
{
  PetscReal eta = 0.04;
  PetscReal a1,a2,a3,a4;
  a1 = 30,a2 = 1, a3 = 5,a4 = 20;
  return -2*eta*(a2+a4*y)/(a1+a2*x+a3*y+a4*x*y);
}
PetscReal pde_bc2(PetscReal x,PetscReal y)
{
   PetscReal eta = 0.04;
   PetscReal a1,a2,a3,a4;
   a1 = 30,a2 = 1,a3 = 5,a4 = 20;
   return -2*eta*(a3+a4*x)/(a1+a2*x+a3*y+a4*x*y);
}
PetscReal soln(PetscReal x,PetscReal y)
{
  PetscReal a1,a2,a3,a4;
  a1 = 30,a2 = 1,a3 = 5,a4 = 20;
  return a1+a2*x+a3*y+a4*x*y;
}
*/

PetscReal soln(PetscReal x,PetscReal y)
{
  PetscReal a1,a2,a3,a4,a5,lambda;
  a1 = 1.3e13;a2 = 1.3e13;a3 = 0;a4 = 0;a5 = 1;
  lambda = 25;
  return a1+a2*(x+1) +a3*y+a4*(x+1)*y+a5*(exp(lambda*x)+exp(-lambda*x))*cos(lambda*y);
}

PetscReal pde_bc1(PetscReal x,PetscReal y)
  {
    PetscReal a2,a4,a5,lambda,eta;
    PetscReal q;
    a2 = 1.3e13;a4 = 0;a5 = 1;
    lambda = 25; eta = 0.04;
    q = soln(x,y);
  return -2*eta*(a2+a4*y+lambda*a5*(exp(lambda*x)-exp(-lambda*x))*cos(lambda*y))/q;
}

PetscReal pde_bc2(PetscReal x,PetscReal y)
{
  PetscReal a3,a4,a5,lambda,eta;
  PetscReal q;
  a3 = 0;a4 = 0;a5 = 1;
  lambda = 25; eta = 0.04;
   q = soln(x,y);
   return -2*eta*(a3+a4*(x+1)-lambda*a5*(exp(lambda*x)+exp(-lambda*x))*sin(lambda*y))/q;
}





