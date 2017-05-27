
static char help[] = "Demonstrates PetscGetVersonNumber().\n\n";

/*T
   Concepts: introduction to PETSc;
   Processors: n
T*/

#include <petscsys.h>
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  char           version[128];
  PetscInt       major,minor,subminor;

  /*
    Every PETSc routine should begin with the PetscInitialize() routine.
    argc, argv - These command line arguments are taken to extract the options
                 supplied to PETSc and options supplied to MPI.
    help       - When PETSc executable is invoked with the option -help,
                 it prints the various options that can be applied at
                 runtime.  The user can use the "help" variable place
                 additional help messages in this printout.
  */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscGetVersion(version,sizeof(version));CHKERRQ(ierr);

  ierr = PetscGetVersionNumber(&major,&minor,&subminor,NULL);CHKERRQ(ierr);
  if (major != PETSC_VERSION_MAJOR) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Library major %d does not equal include %d",(int)major,PETSC_VERSION_MAJOR);
  if (minor != PETSC_VERSION_MINOR) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Library minor %d does not equal include %d",(int)minor,PETSC_VERSION_MINOR);
  if (subminor != PETSC_VERSION_SUBMINOR) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Library subminor %d does not equal include %d",(int)subminor,PETSC_VERSION_SUBMINOR);

  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:

TEST*/
