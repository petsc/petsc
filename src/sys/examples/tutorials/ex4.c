#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex4.c,v 1.6 1999/05/18 14:37:04 bsmith Exp bsmith $";
#endif

static char help[] = "Prints loadable objects from dynamic library.\n\n";

/*T
   Concepts: Dynamic libraries;
   Routines: PetscInitialize(); PetscPrintf(); PetscFinalize();
   Processors: n
T*/
 
#include "petsc.h"
#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int  ierr,flag;
  char *string,filename[256];
  void *handle;

  /*
    Every PETSc routine should begin with the PetscInitialize() routine.
    argc, argv - These command line arguments are taken to extract the options
                 supplied to PETSc and options supplied to MPI.
    help       - When PETSc executable is invoked with the option -help, 
                 it prints the various options that can be applied at 
                 runtime.  The user can use the "help" variable place
                 additional help messages in this printout.
  */
  ierr = PetscInitialize(&argc,&argv,(char *)0,help);CHKERRA(ierr);

  ierr = OptionsGetString(PETSC_NULL,"-library",filename,256,&flag);CHKERRA(ierr);
  if (!flag) {
    SETERRA(1,1,"Must indicate library name with -library");
  }

#if defined(USE_DYNAMIC_LIBRARIES)
  ierr = DLLibraryOpen(PETSC_COMM_WORLD,filename,&handle);CHKERRA(ierr);
  ierr = DLLibraryGetInfo(handle,"Contents",&string);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Contents:%s\n",string);CHKERRA(ierr);
  ierr = DLLibraryGetInfo(handle,"Authors",&string);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Authors:%s\n",string);CHKERRA(ierr);
  ierr = DLLibraryGetInfo(handle,"Version",&string);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Version:%s\n",string);CHKERRA(ierr);
#else
  /* just forces string and handle to be used so there are no compiler warnings */
  string = "No dynamic libraries used";
  handle = (void *) string;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s\n",string);CHKERRA(ierr);
  flag = !PetscStrcmp(string,"Never will happen");
  if (flag) {
    ierr = PetscObjectDestroy((PetscObject)handle);CHKERRA(ierr);
  }
#endif

  ierr = PetscFinalize();CHKERRA(ierr);
  return 0;
}
