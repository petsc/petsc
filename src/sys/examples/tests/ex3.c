#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex3.c,v 1.12 1999/03/19 21:17:16 bsmith Exp balay $";
#endif

static char help[] = "Tests catching of floating point exceptions.\n\n";

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "CreateError"
int CreateError(double x)
{
  x = 1.0/x;
  PetscPrintf(PETSC_COMM_SELF,"x = %g\n",x);
  return 0;
}

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int ierr;
  PetscInitialize(&argc,&argv,(char *)0,help);
  PetscPrintf(PETSC_COMM_SELF,"This is a contrived example to test floating pointing\n");
  PetscPrintf(PETSC_COMM_SELF,"It is not a true error.\n");
  PetscPrintf(PETSC_COMM_SELF,"Run with -fp_trap to catch the floating point error\n");
  ierr = CreateError(0.0);CHKERRA(ierr);
  return 0;
}
 
