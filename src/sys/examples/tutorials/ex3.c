
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
  int ierr;
  PetscInitialize(&argc,&argv,0,0);
  if (OptionsHasName(0,"-help")) fprintf(stderr,help);
  fprintf(stderr,"This is a contrived example to test floating pointing\n");
  fprintf(stderr,"It is not a true error.\n");
  fprintf(stderr,"Run with -fp_trap to catch the floating point error\n");
  fflush(stderr);
  ierr = CreateError(0.0); CHKERRA(ierr);
  return 0;
}
 
