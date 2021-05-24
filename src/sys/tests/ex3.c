
static char help[] = "Tests catching of floating point exceptions.\n\n";

#include <petscsys.h>

int CreateError(PetscReal x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  x    = 1.0/x;
  ierr = PetscPrintf(PETSC_COMM_SELF,"x = %g\n",(double)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscPrintf(PETSC_COMM_SELF,"This is a contrived example to test floating pointing\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"It is not a true error.\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Run with -fp_trap to catch the floating point error\n");CHKERRQ(ierr);
  ierr = CreateError(0.0);CHKERRQ(ierr);
  return 0;
}

/*

    Because this example may produce different output on different machines we filter out everything.
    This makes the test ineffective but currently we don't have a good way to know which machines should handle
    the floating point exceptions properly.

*/
/*TEST

   test:
      args: -fp_trap -error_output_stdout
      filter: Error: true

TEST*/
