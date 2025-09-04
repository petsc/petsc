static char help[] = "Tests MPIU_Allreduce() for large count.\n";
#include <petscsys.h>

int main(int argc, char **args)
{
  PetscBool same = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &same, PETSC_INT_MAX - 100, MPI_C_BOOL, MPI_LAND, PETSC_COMM_WORLD));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     requires: !defined(PETSC_HAVE_MPI_LARGE_COUNT) defined(PETSC_HAVE_64_BIT_INDICES)
     args: -petsc_ci_portable_error_output -error_output_stdout
     filter: grep -E "(PETSC ERROR)" | grep -E "(Error Created|CreateError\(\)|main\(\))"

TEST*/
