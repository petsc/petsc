#include "appctx.h"
 
/* have these functions take appctx->equations as a parameter and read in the necessary stuff, 
including a flag e.g. cavity-flag */

/* forcing functions */
 double g(double x, double y){ 
  return 0; } 
 double f(double x, double y){
   return 0; }

/* Bc1, bc2 are the solutions for u,v */
/* bc3 is for presure */
double bc1(AppEquations *equations)
{  
  double mu, R, x, y;
  R = 1/equations->eta;
  x = equations->xval;
  y = equations->yval;
  if(equations->cylinder_flag){
    mu = .5*R + sqrt(.25*R*R + 4*PETSC_PI*PETSC_PI);
    return 1 - exp(-mu*x)*cos(2*PETSC_PI*y);
  }
  if(equations->cavity_flag){
    return 0;} 
  if(equations->parabolic_flag){
    return  equations->yval*(1-equations->yval)*10*0.5*R;
    /*   y(1-y)R(deltaP/2*L) */
  }
  else
    return 0;
}

double bc2(AppEquations *equations)
{  
  double mu, R, x, y;
  R = 1/equations->eta;
  x = equations->xval;
  y = equations->yval;
  if(equations->cylinder_flag){
    mu = .5*R + sqrt(.25*R*R + 4*PETSC_PI*PETSC_PI);
    return -mu/(2*PETSC_PI)*exp(-mu*x)*sin(2*PETSC_PI*y);
  }
  if(equations->cavity_flag){
    return 1.0102*(1-equations->xval);}

  if(equations->parabolic_flag){
    return 0; }
  else
      return 0;
}

double bc3(AppEquations *equations)
{
  double mu, R, x, y;
  R = 1/equations->eta;
  x = equations->xval;
  y = equations->yval;
 if(equations->cylinder_flag){
    mu = .5*R + sqrt(.25*R*R + 4*PETSC_PI*PETSC_PI);
    return .5*(1-exp(-2*mu*x));}
 if(equations->parabolic_flag){
  /* p(x) = p1 - deltap/L*x */
   return 10 - 10*equations->xval;}
else
  return 0;
}


