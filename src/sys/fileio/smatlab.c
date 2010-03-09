#define PETSC_DLL

#include "petscsys.h"

#undef __FUNCT__
#define __FUNCT__ "PetscStartMatlab"	
/*@C
    PetscStartMatlab - starts up Matlab with a Matlab script

    Collective on MPI_Comm, but only processor zero in the communicator does anything

    Input Parameters:
+     comm - MPI communicator
.     machine - optional machine to run Matlab on
-     script - name of script (without the .m)

    Output Parameter:
.     fp - a file pointer returned from PetscPOpen()

    Level: intermediate

    Notes: 
     This overwrites your matlab/startup.m file

     The script must be in your Matlab path or current directory

     Assumes that all machines share a common file system

.seealso: PetscPOpen(), PetscPClose()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscStartMatlab(MPI_Comm comm,const char machine[],const char script[],FILE **fp)
{
  PetscErrorCode ierr;
  FILE           *fd;
  char           command[512];
#if defined(PETSC_HAVE_UCBPS) && defined(PETSC_HAVE_POPEN)
  char           buf[1024],*found;
  PetscMPIInt    rank;
#endif

  PetscFunctionBegin;

#if defined(PETSC_HAVE_UCBPS) && defined(PETSC_HAVE_POPEN)
  /* check if Matlab is not already running */
  ierr = PetscPOpen(comm,machine,"/usr/ucb/ps -ugxww | grep matlab | grep -v grep","r",&fd);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    found = fgets(buf,1024,fd);
  }
  ierr = MPI_Bcast(&found,1,MPI_CHAR,0,comm);CHKERRQ(ierr);
  ierr = PetscPClose(comm,fd);CHKERRQ(ierr);
  if (found) PetscFunctionReturn(0);
#endif

  if (script) {
    /* the remote machine won't know about current directory, so add it to Matlab path */
    /* the extra \" are to protect possible () in the script command from the shell */
    sprintf(command,"echo \"delete ${HOMEDIRECTORY}/matlab/startup.m ; path(path,'${WORKINGDIRECTORY}'); %s  \" > ${HOMEDIRECTORY}/matlab/startup.m",script);
#if defined(PETSC_HAVE_POPEN)
    ierr = PetscPOpen(comm,machine,command,"r",&fd);CHKERRQ(ierr);
    ierr = PetscPClose(comm,fd);CHKERRQ(ierr);
#endif
  }
#if defined(PETSC_HAVE_POPEN)
  ierr = PetscPOpen(comm,machine,"xterm -display ${DISPLAY} -e matlab -nosplash","r",fp);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

