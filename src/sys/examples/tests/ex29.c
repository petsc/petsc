
static char help[] = "Tests PetscInt64Mult()\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       a = 2009,b = 5612,result,tresult;
  PetscInt64     r64;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr    = PetscIntMultError(a,b,&result);CHKERRQ(ierr);
  a       = PETSC_MPI_INT_MAX-22,b = PETSC_MPI_INT_MAX/22;
  r64     = PetscInt64Mult(a,b);
  tresult = PetscIntMultTruncate(a,b);
  ierr    = PetscIntMultError(a,b,&result);CHKERRQ(ierr);
  ierr    = ierr = PetscFinalize();
  return ierr;
}

