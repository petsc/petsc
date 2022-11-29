
static char help[] = "Demonstrates PetscGetVersonNumber().\n\n";

#include <petscsys.h>
int main(int argc, char **argv)
{
  char     version[128];
  PetscInt major, minor, subminor;

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
  PetscCall(PetscGetVersion(version, sizeof(version)));

  PetscCall(PetscGetVersionNumber(&major, &minor, &subminor, NULL));
  PetscCheck(major == PETSC_VERSION_MAJOR, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Library major %d does not equal include %d", (int)major, PETSC_VERSION_MAJOR);
  PetscCheck(minor == PETSC_VERSION_MINOR, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Library minor %d does not equal include %d", (int)minor, PETSC_VERSION_MINOR);
  PetscCheck(subminor == PETSC_VERSION_SUBMINOR, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Library subminor %d does not equal include %d", (int)subminor, PETSC_VERSION_SUBMINOR);

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
