
/* Program usage:  mpiexec ex1 [-help] [all PETSc options] */

static char help[] = "Introductory example that illustrates printing.\n\n";

/*T
   Concepts: introduction to PETSc;
   Concepts: printing^in parallel
   Processors: n
T*/
 
#include <petscsys.h>
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;

  /*
    Every PETSc routine should begin with the PetscInitialize() routine.
    argc, argv - These command line arguments are taken to extract the options
                 supplied to PETSc and options supplied to MPI.
    help       - When PETSc executable is invoked with the option -help, 
                 it prints the various options that can be applied at 
                 runtime.  The user can use the "help" variable place
                 additional help messages in this printout.
  */
  ierr = PetscInitialize(&argc,&argv,(char *)0,help);CHKERRQ(ierr);

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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of processors = %d, rank = %d\n",size,rank);CHKERRQ(ierr);

  /*
    Here a barrier is used to separate the two program states.
  */
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);

  /*
    Here we simply use PetscPrintf() with the communicator PETSC_COMM_SELF,
    where each process is considered separately and prints independently
    to the screen.  Thus, the output from different processes does not
    appear in any particular order.
  */

  ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Jumbled Hello World\n",rank);CHKERRQ(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary).  See PetscFinalize()
     manpage for more information.
  */
  ierr = PetscFinalize();
  return 0;
}
