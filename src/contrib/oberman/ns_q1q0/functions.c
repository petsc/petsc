
/* functions.c */
#include <math.h>

/* forcing fucntions */
 double g(double x, double y){ 
  return 0; } 
 double f(double x, double y){
   return 0; }
/* Bc1, bc2 are the solutions for u,v */
/* bc3 is for presure */
double bc1(double x, double y)
{  
return 0;
/*   return y*(1-y)*10*10/2;     */
/*   y(1-y)R(deltaP/2*L) */
}
double bc2(double x, double y)
  {return 1-x;}
double bc3(double x, double y)
{
  /* p(x) = p1 - deltap/L*x */
  return 10 - 10*x;
}

