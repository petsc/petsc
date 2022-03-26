
static char help[] = "Introductory example that illustrates printing.\n\n";

/*T
   Concepts: introduction to PETSc;
   Concepts: printing^in parallel
   Processors: n
T*/

#include <petscsys.h>
int main(int argc,char **argv)
{
  PetscMPIInt    rank,size;

  /*
    Every PETSc program should begin with the PetscInitialize() routine.
    argc, argv - These command line arguments are taken to extract the options
                 supplied to PETSc and options supplied to MPI.
    help       - When PETSc executable is invoked with the option -help,
                 it prints the various options that can be applied at
                 runtime.  The user can use the "help" variable to place
                 additional help messages in this printout.
  */
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  /*
     The following MPI calls return the number of processes
     being used and the rank of this process in the group.
   */
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /*
     Here we would like to print only one message that represents
     all the processes in the group.  We use PetscPrintf() with the
     communicator PETSC_COMM_WORLD.  Thus, only one message is
     printed representng PETSC_COMM_WORLD, i.e., all the processors.
  */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Number of processors = %d, rank = %d\n",size,rank));

  /*
    Here a barrier is used to separate the two program states.
  */
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

  /*
    Here we simply use PetscPrintf() with the communicator PETSC_COMM_SELF,
    where each process is considered separately and prints independently
    to the screen.  Thus, the output from different processes does not
    appear in any particular order.
  */

  PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d] Jumbled Hello World\n",rank));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).  See PetscFinalize()
     manpage for more information.
  */
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
