
static char help[] = "Demonstrates call PETSc and Chombo in the same program.\n\n";

#include <petscsys.h>
#include "Box.H"

int main(int argc,char **argv)
{

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
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  Box::Box *nb = new Box::Box();
  delete nb;

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: chombo

   test:

TEST*/
