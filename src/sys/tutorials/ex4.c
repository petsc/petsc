static char help[] = "Introductory example that illustrates running PETSc on a subset of processes.\n\n";

#include <petscsys.h>

int main(int argc, char *argv[])
{
  PetscMPIInt rank, size;

  /* We must call MPI_Init() first, making us, not PETSc, responsible for MPI */
  PetscCallMPI(MPI_Init(&argc, &argv));
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscCall(PetscElementalInitializePackage());
#endif
  /* We can now change the communicator universe for PETSc */
  PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_split(MPI_COMM_WORLD, rank % 2, 0, &PETSC_COMM_WORLD));

  /*
    Every PETSc routine should begin with the PetscInitialize() routine.
    argc, argv - These command line arguments are taken to extract the options
                 supplied to PETSc and options supplied to MPI.
    help       - When PETSc executable is invoked with the option -help,
                 it prints the various options that can be applied at
                 runtime.  The user can use the "help" variable place
                 additional help messages in this printout.
  */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  /*
     The following MPI calls return the number of processes
     being used and the rank of this process in the group.
   */
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /*
     Here we would like to print only one message that represents
     all the processes in the group.  We use PetscPrintf() with the
     communicator PETSC_COMM_WORLD.  Thus, only one message is
     printed representng PETSC_COMM_WORLD, i.e., all the processors.
  */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of processors = %d, rank = %d\n", size, rank));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).  See PetscFinalize()
     manpage for more information.
  */
  PetscCall(PetscFinalize());
  PetscCallMPI(MPI_Comm_free(&PETSC_COMM_WORLD));
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscCall(PetscElementalFinalizePackage());
#endif
  /* Since we initialized MPI, we must call MPI_Finalize() */
  PetscCallMPI(MPI_Finalize());
  return 0;
}

/*TEST

   test:
      nsize: 5
      args: -options_left no
      filter: sort -b | grep -v saws_port_auto_selectcd
      filter_output: sort -b

TEST*/
