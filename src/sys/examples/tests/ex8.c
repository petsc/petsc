#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex8.c,v 1.4 1997/04/10 00:01:45 bsmith Exp balay $";
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
  
  PetscPrintf(PETSC_COMM_SELF,"y = %f \n",y);
  PetscFree(x);
  PetscFinalize();
  return 0;
}
 
