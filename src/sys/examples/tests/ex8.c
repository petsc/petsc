#ifndef lint
static char vcid[] = "$Id: ex8.c,v 1.1 1997/02/07 23:25:00 bsmith Exp balay $";
#endif

static char *help = "Tests the option -trmalloc_nan which initializes the memory \n\
allocated by PetscMalloc() with NaNs\n\
Use Options: -tr_malloc_nan -fp_trap\n\n";


#include "petsc.h"
int main(int argc,char **args)
{

  Scalar *x,y=0;

  PetscInitialize(&argc,&args,(char *)0,help);

  x = (Scalar*)PetscMalloc(1*sizeof(Scalar)); CHKPTRA(x);

  y += x[0];
  PetscFree(x);
  PetscFinalize();
  return 0;
}
 
