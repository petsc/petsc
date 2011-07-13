
/* Program usage:  mpiexec ex1 [-help] [all PETSc options] */

static char help[] = "Demonstrates using PetscWebServe().\n\n";

/*T
   Concepts: introduction to PETSc;
   Concepts: printing^in parallel
   Processors: n
T*/
 
#include <petscsys.h>
int main(int argc,char **argv)
{
  PetscErrorCode ierr;

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

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting up PetscWebServe()\n");CHKERRQ(ierr);
#if defined(PETSC_USE_SERVER)
  ierr = PetscWebServe(PETSC_COMM_WORLD,8000);CHKERRQ(ierr);
  while (1) {;}
#endif

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
