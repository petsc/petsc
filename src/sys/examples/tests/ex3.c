
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
  if (OptionsHasName(0,"-help")) fprintf(stdout,help);
  fprintf(stdout,"This is a contrived example to test floating pointing\n");
  fprintf(stdout,"It is not a true error.\n");
  fprintf(stdout,"Run with -fp_trap to catch the floating point error\n");
  fflush(stdout);
  ierr = CreateError(0.0); CHKERRA(ierr);
  return 0;
}
 
