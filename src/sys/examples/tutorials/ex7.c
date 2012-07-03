
/* Program usage:  mpiexec ex1 [-help] [all PETSc options] */

static char help[] = "Demonstrates using PetscWebServe().\nRun with -ams_publish_objects\n\n";

/*T
   Concepts: introduction to PETSc;
   Concepts: printing^in parallel
   Processors: n
T*/
 
#include <petscsys.h>
#include <petscksp.h>
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscRandom    rand1,rand2;

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
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand1);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand1);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand2);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand2);CHKERRQ(ierr);
#if defined(PETSC_USE_SERVER)
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Starting up PetscWebServe()\n");CHKERRQ(ierr);
  ierr = PetscWebServe(PETSC_COMM_WORLD,PETSC_DEFAULT);CHKERRQ(ierr); 
  while (1) {;}
#endif
  ierr = PetscRandomDestroy(&rand1);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand2);CHKERRQ(ierr);
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
