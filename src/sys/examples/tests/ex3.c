
static char help[] = 
"This example tests catching of floating point exceptions.\n\n";

#include "petsc.h"
#include <stdio.h>

int CreateError(double x)
{
  x = 1.0/x;
  printf("x = %g\n",x);
  return 0;
}

int main(int argc,char **argv)
{
  PetscInitialize(&argc,&argv,0,0);
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,help);
  fprintf(stderr,"This is a contrived example to test floating pointing\n");
  fprintf(stderr,"It is not a true error.\n");
  fflush(stderr);
  CreateError(0.0);
  return 0;
}
 
