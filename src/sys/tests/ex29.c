
static char help[] = "Tests PetscInt64Mult()\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       a = 2009,b = 5612,result,tresult;
  PetscInt64     r64;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscIntMultError(a,b,&result));
  a       = PETSC_MPI_INT_MAX-22,b = PETSC_MPI_INT_MAX/22;
  r64     = PetscInt64Mult(a,b);
  tresult = PetscIntMultTruncate(a,b);
  CHKERRQ(PetscIntMultError(a,b,&result));
  ierr    = CHKERRQ(PetscFinalize());
  return 0;
}
