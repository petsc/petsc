static char help[] = "Tests MPIU_Allreduce() for overflow.\n";
#include <petscsys.h>

int main(int argc, char **args)
{
  PetscInt same = PETSC_INT_MAX;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &same, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     TODO: Cannot run CI test in parallel to produce clean output
     requires: !defined(PETSC_USE_64_BIT_INDICES)
     nsize: 2
     args: -petsc_ci_portable_error_output -error_output_stdout
     filter: grep -E "(PETSC ERROR)" | grep -E "(Error Created|CreateError\(\)|main\(\))"

TEST*/
