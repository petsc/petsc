
static char help[] = "Tests calling PetscOptionsSetValue() before PetscInitialize()\n\n";

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
  ierr = PetscOptionsSetValue(NULL,"-no_signal_handler","true");
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of processors = %d, rank = %d\n",size,rank);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
