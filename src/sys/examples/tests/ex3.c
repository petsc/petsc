
static char help[] = 
"This example tests catching of floating point exceptions.\n\n";

#include "petsc.h"
#include <stdio.h>

int CreateError(double x)
{
  x = 1.0/x;
  MPIU_printf(MPI_COMM_SELF,"x = %g\n",x);
  return 0;
}

int main(int argc,char **argv)
{
  int ierr;
  PetscInitialize(&argc,&argv,0,0,help);
  MPIU_printf(MPI_COMM_SELF,"This is a contrived example to test floating pointing\n");
  MPIU_printf(MPI_COMM_SELF,"It is not a true error.\n");
  MPIU_printf(MPI_COMM_SELF,"Run with -fp_trap to catch the floating point error\n");
  ierr = CreateError(0.0); CHKERRA(ierr);
  return 0;
}
 
