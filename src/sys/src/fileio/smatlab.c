/* $Id: smatlab.c,v 1.1 2000/01/05 21:37:32 bsmith Exp bsmith $ */

#include "petsc.h"
#include "sys.h"

#undef __FUNC__
#define __FUNC__ "PetscStartMatlab"
/*@C
    PetscStartMatlab - starts up Matlab with a Matlab script

    Not collective, but only processor zero in the communicator does anything

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
int PetscStartMatlab(MPI_Comm comm,char *machine,char *script,FILE **fp)
{
  int  ierr;
  char command[512];

  PetscFunctionBegin;
  /* the extra \" are to protect possible () in the script command from the shell */

  /* the remote machine won't know about current directory, so add it to Matlab path */
  ierr = PetscStrcpy(command,"echo \"path(path,'$WORKINGDIRECTORY'); ");CHKERRQ(ierr);
  ierr = PetscStrcat(command,script);CHKERRQ(ierr);
  ierr = PetscStrcat(command,"\" > $HOMEDIRECTORY/matlab/startup.m");CHKERRQ(ierr);
  ierr = PetscPOpen(comm,PETSC_NULL,command,"r",fp);CHKERRQ(ierr);
  ierr = PetscFClose(comm,*fp);CHKERRQ(ierr);

  ierr = PetscPOpen(comm,machine,"xterm -display $DISPLAY -e matlab -nosplash","r",fp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

