
static char help[] = "Tests PetscIntMult64bit()\n";

#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       a = 2009,b = 5612,result,tresult;
  Petsc64bitInt  r64;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr    = PetscIntMultError(a,b,&result);CHKERRQ(ierr);
  a       = PETSC_MPI_INT_MAX-22,b = PETSC_MPI_INT_MAX/22;
  r64     = PetscIntMult64bit(a,b);
  tresult = PetscIntMultTruncate(a,b);
  ierr    = PetscIntMultError(a,b,&result);CHKERRQ(ierr);
  ierr    = PetscFinalize();
  return 0;
}

