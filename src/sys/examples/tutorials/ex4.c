
static char help[] = "Prints loadable objects from dynamic library.\n\n";

/*T
   Concepts: dynamic libraries;
   Processors: n
T*/
 
#include "petsc.h"
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  int        ierr;
  PetscTruth flg;
  const char *string;
  char       filename[PETSC_MAX_PATH_LEN];
  void       *handle;

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

  ierr = PetscOptionsGetString(PETSC_NULL,"-library",filename,256,&flg);CHKERRQ(ierr);
  if (!flg) {
    SETERRQ(1,"Must indicate library name with -library");
  }

#if defined(USE_DYNAMIC_LIBRARIES)
  ierr = PetscDLLibraryOpen(PETSC_COMM_WORLD,filename,&handle);CHKERRQ(ierr);
  ierr = PetscDLLibraryGetInfo(handle,"Contents",&string);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Contents:%s\n",string);CHKERRQ(ierr);
  ierr = PetscDLLibraryGetInfo(handle,"Authors",&string);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Authors:%s\n",string);CHKERRQ(ierr);
  ierr = PetscDLLibraryGetInfo(handle,"Version",&string);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Version:%s\n",string);CHKERRQ(ierr);
#else
  /* just forces string and handle to be used so there are no compiler warnings */
  string = "No dynamic libraries used";
  handle = (void*)string;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s\n",string);CHKERRQ(ierr);
  ierr = PetscStrcmp(string,"Never will happen",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscObjectDestroy((PetscObject)handle);CHKERRQ(ierr);
  }
#endif

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
