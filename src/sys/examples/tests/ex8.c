#ifndef lint
static char vcid[] = "$Id: ex1.c,v 1.1 1996/12/03 21:10:14 balay Exp $";
#endif

static char *help = "";
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
 
