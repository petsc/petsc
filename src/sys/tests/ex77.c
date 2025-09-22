static char help[] = "Tests MPIU_Allreduce() for mis-use.\n";
#include <petscsys.h>

int main(int argc, char **args)
{
  PetscMPIInt rank;
  PetscBool   same = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (!rank) PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &same, 1, MPI_C_BOOL, MPI_LAND, PETSC_COMM_WORLD));
  else PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &same, 1, MPI_C_BOOL, MPI_LAND, PETSC_COMM_WORLD));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     requires: defined(PETSC_USE_DEBUG) !defined(PETSCTEST_VALGRIND) !defined(PETSC_HAVE_SANITIZER)
     args: -petsc_ci_portable_error_output -error_output_stdout
     nsize: 2
     filter: grep -E "(PETSC ERROR)" | grep -E "(Error Created|CreateError\(\)|main\(\))"

TEST*/
