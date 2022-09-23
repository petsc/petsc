
static char help[] = "Tests calling PetscOptionsSetValue() before PetscInitialize()\n\n";

#include <petscsys.h>
int main(int argc, char **argv)
{
  PetscMPIInt rank, size;

  /*
    Every PETSc routine should begin with the PetscInitialize() routine.
    argc, argv - These command line arguments are taken to extract the options
                 supplied to PETSc and options supplied to MPI.
    help       - When PETSc executable is invoked with the option -help,
                 it prints the various options that can be applied at
                 runtime.  The user can use the "help" variable place
                 additional help messages in this printout.

    Since when PetscInitialize() returns with an error the PETSc data structures
    may not be set up hence we cannot call PetscCall() hence directly return the error code.

    Since PetscOptionsSetValue() is called before the PetscInitialize() we cannot call
    PetscCall() on the error code and just return it directly.
  */
  PetscCall(PetscOptionsSetValue(NULL, "-no_signal_handler", "true"));
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of processors = %d, rank = %d\n", size, rank));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      requires: defined(PETSC_USE_LOG)
      nsize: 2
      args: -options_view -get_total_flops
      filter: grep -E -v "(cuda_initialize|Total flops|options_left)"

TEST*/
