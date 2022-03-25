
#include <petscsys.h>

/*@C
    PetscStartMatlab - starts up MATLAB with a MATLAB script

    Logically Collective, but only processor zero in the communicator does anything

    Input Parameters:
+     comm - MPI communicator
.     machine - optional machine to run MATLAB on
-     script - name of script (without the .m)

    Output Parameter:
.     fp - a file pointer returned from PetscPOpen()

    Level: intermediate

    Notes:
     This overwrites your matlab/startup.m file

     The script must be in your MATLAB path or current directory

     Assumes that all machines share a common file system

.seealso: PetscPOpen(), PetscPClose()
@*/
PetscErrorCode  PetscStartMatlab(MPI_Comm comm,const char machine[],const char script[],FILE **fp)
{
  FILE           *fd;
  char           command[512];
#if defined(PETSC_HAVE_UCBPS) && defined(PETSC_HAVE_POPEN)
  char           buf[1024],*found;
  PetscMPIInt    rank;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_UCBPS) && defined(PETSC_HAVE_POPEN)
  /* check if MATLAB is not already running */
  PetscCall(PetscPOpen(comm,machine,"/usr/ucb/ps -ugxww | grep matlab | grep -v grep","r",&fd));
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  if (rank == 0) found = fgets(buf,1024,fd);
  PetscCallMPI(MPI_Bcast(&found,1,MPI_CHAR,0,comm));
  PetscCall(PetscPClose(comm,fd));
  if (found) PetscFunctionReturn(0);
#endif

  if (script) {
    /* the remote machine won't know about current directory, so add it to MATLAB path */
    /* the extra \" are to protect possible () in the script command from the shell */
    sprintf(command,"echo \"delete ${HOMEDIRECTORY}/matlab/startup.m ; path(path,'${WORKINGDIRECTORY}'); %s  \" > ${HOMEDIRECTORY}/matlab/startup.m",script);
#if defined(PETSC_HAVE_POPEN)
    PetscCall(PetscPOpen(comm,machine,command,"r",&fd));
    PetscCall(PetscPClose(comm,fd));
#endif
  }
#if defined(PETSC_HAVE_POPEN)
  PetscCall(PetscPOpen(comm,machine,"xterm -display ${DISPLAY} -e matlab -nosplash","r",fp));
#endif
  PetscFunctionReturn(0);
}
