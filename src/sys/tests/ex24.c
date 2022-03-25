static char help[] = "Tests %d and %g formatting\n";
#include <petscsys.h>

int main(int argc,char **argv)
{

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"A string followed by integer %d\n",22));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"A string followed by double %5g another %g\n",23.2,11.3));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"and then an int %d\n",30));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     diff_args: -j

TEST*/
