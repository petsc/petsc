#ifndef lint
static char vcid[] = "$Id: ex4.c,v 1.12 1996/02/13 23:28:56 bsmith Exp bsmith $";
#endif

static char help[] = 
"Demonstrates how users can augment the PETSc profiling by\n\
inserting their own event logging.  Run this program with one of the\n\
following options to generate logging information:  -log, -log_summary,\n\
-log_all.  The PETSc routines automatically log event times and flops,\n\
so this monitoring is intended solely for users to employ in application\n\
codes.  Note that the code must be compiled with the flag -DPETSC_LOG\n\
(the default) to activate logging.\n\n";

#include "petsc.h"
#include <stdio.h>

int main(int argc,char **argv)
{
  int i, imax=10000, icount;
  int USER_EVENT;

  PetscInitialize(&argc,&argv,0,0,help);

  PLogEventRegister(&USER_EVENT,"User event      ","Red");
  PLogEventBegin(USER_EVENT,0,0,0,0);
    icount = 0;
    for (i=0; i<imax; i++) icount++;
    PLogFlops(imax);
  PLogEventEnd(USER_EVENT,0,0,0,0);
  PetscFinalize();
  return 0;
}
 
