
static char help[] = "Demonstrates PetscFileRetrieve().\n\n";

/*T
   Concepts: introduction to PETSc;
   Concepts: printing^in parallel
   Processors: n
T*/

#include <petscsys.h>
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscBool      found;
  char           localname[PETSC_MAX_PATH_LEN];
  const char     url[] = "https://www.mcs.anl.gov/petsc/index.html";

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
  ierr = PetscFileRetrieve(PETSC_COMM_WORLD,url,localname,PETSC_MAX_PATH_LEN,&found);CHKERRQ(ierr);
  if (found) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Successfully download file %s\n",localname);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Unable to download url %s\n",url);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     requires: define(PETSC_HAVE_POPEN)

TEST*/
