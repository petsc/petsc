static char help[] = "Tests %D and %g formatting\n";
#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"A string followed by integer %d\n",22);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"A string followed by double %5g another %g\n",23.2,11.3);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"and then an int %d\n",30);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     diff_args: -j

TEST*/
