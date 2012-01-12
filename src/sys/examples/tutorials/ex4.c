/* Program usage:  mpiexec -n 2 ex1 [-help] [all PETSc options] */

static char help[] = "Introductory example that illustrates running PETSc on a subset of processes.\n\n";

/*T
   Concepts: introduction to PETSc;
   Concepts: process^subset set PETSC_COMM_WORLD
   Processors: 2
T*/
 #include <petscsys.h>

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  PetscMPIInt    rank, size;

  /* We must call MPI_Init() first, making us, not PETSc, responsible for MPI */
  ierr = MPI_Init(&argc, &argv);CHKERRQ(ierr);

  /* We can now change the communicator universe for PETSc */
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_split(MPI_COMM_WORLD, rank%2, 0, &PETSC_COMM_WORLD);CHKERRQ(ierr);

  /*
    Every PETSc routine should begin with the PetscInitialize() routine.
    argc, argv - These command line arguments are taken to extract the options
                 supplied to PETSc and options supplied to MPI.
    help       - When PETSc executable is invoked with the option -help, 
                 it prints the various options that can be applied at 
                 runtime.  The user can use the "help" variable place
                 additional help messages in this printout.
  */
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);

  /* 
     The following MPI calls return the number of processes
     being used and the rank of this process in the group.
   */
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* 
     Here we would like to print only one message that represents
     all the processes in the group.  We use PetscPrintf() with the 
     communicator PETSC_COMM_WORLD.  Thus, only one message is
     printed representng PETSC_COMM_WORLD, i.e., all the processors.
  */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of processors = %d, rank = %d\n", size, rank);CHKERRQ(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary).  See PetscFinalize()
     manpage for more information.
  */
  ierr = PetscFinalize();

  /* Since we initialized MPI, we must call MPI_Finalize() */
  ierr = MPI_Finalize();
  return 0;
}
