static char help[] = "Error handling for destroying PETSC_VIEWER_STDOUT_SELF.\n";

#include <petscsys.h>
#include <petscviewer.h>

int main(int argc, char **argv)
{
  PetscViewer viewer;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  viewer = PETSC_VIEWER_STDOUT_SELF;
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     requires: !defined(PETSCTEST_VALGRIND)
     args: -petsc_ci_portable_error_output -error_output_stdout
     filter: grep -E -v "(memory block|leaked context|not freed before MPI_Finalize|Could be the program crashed|PETSc Option Table entries|source: environment)"

TEST*/
