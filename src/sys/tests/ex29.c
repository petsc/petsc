
static char help[] = "Tests PetscInt64Mult()\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscInt       a = 2009,b = 5612,result,tresult;
  PetscInt64     r64;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscIntMultError(a,b,&result));
  a       = PETSC_MPI_INT_MAX-22,b = PETSC_MPI_INT_MAX/22;
  r64     = PetscInt64Mult(a,b);
  tresult = PetscIntMultTruncate(a,b);
  PetscCall(PetscIntMultError(a,b,&result));
  PetscCall(PetscFinalize());
  return 0;
}
