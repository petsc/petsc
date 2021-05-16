
static char help[] = "Tests the signal handler.\n";

#include <petscsys.h>

int CreateError(int n)
{
  PetscErrorCode ierr;
  PetscReal      *x = 0;
  if (!n) {x[0] = 100.; return 0;}
  ierr = CreateError(n-1);CHKERRQ(ierr);
  return 0;
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscFPrintf(PETSC_COMM_WORLD,stdout,"Demonstrates how PETSc can trap error interrupts\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_WORLD,stdout,"The error below is contrived to test the code!\n");CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
  ierr = CreateError(5);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     args: -error_output_stdout
     filter: egrep "(Caught signal number 11 SEGV|Caught signal number 4 Illegal)" | wc -l
     TODO:  Does not always produce exactly expected output on all systems for all runs

TEST*/
