/*$Id: select.c,v 1.1 2000/04/20 03:35:46 bsmith Exp bsmith $*/
#include "petsc.h"         /*I  "petsc.h"  I*/
#include "sys.h"           /*I  "sys.h"  I*/

#undef __FUNC__  
#define __FUNC__ /*<a name="PetscPopUpSelect"></a>*/"PetscPopUpSelect"
/*@C
     PetscPopUpSelect - Pops up a windows with a list of choices; allows one to be chosen

     Collective on MPI_Comm

     Input Parameters:
+    comm - MPI communicator, all processors in communicator must call this but input 
            from first communicator is the only one that is used
.    machine - location to run popup program or PETSC_NULL
.    n - number of choices
-    choices - array of strings

     Output Parameter:
.    choice - integer indicating which one was selected

     Level: developer

     Notes:
       Uses DISPLAY variable or -display option to deteremine where it opens the window

.keywords: architecture, machine     
@*/
int PetscPopUpSelect(MPI_Comm comm,char *machine,int n,char **choices,int *choice)
{
  int  i,ierr,rank,rows = n + 3,cols = 0,len;
  char buffer[2048],display[128],geometry[64];
  FILE *fp;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    ierr = PetscStrlen(choices[i],&len);CHKERRQ(ierr);
    cols = PetscMax(cols,len);
  }
  sprintf(geometry,"-geometry %dx%d ",rows,cols);
  ierr = PetscStrcpy(buffer,"xterm -display ");
  ierr = PetscGetDisplay(display,128);CHKERRQ(ierr);
  ierr = PetscStrcat(buffer,geometry);CHKERRQ(ierr);
  ierr = PetscStrcat(buffer,display);CHKERRQ(ierr);
  ierr = PetscStrcat(buffer," -e ${PETSC_DIR}/bin/popup ");CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscStrcat(buffer,"\"");CHKERRQ(ierr);
    ierr = PetscStrcat(buffer,choices[i]);CHKERRQ(ierr);
    ierr = PetscStrcat(buffer,"\" ");CHKERRQ(ierr);
  }
  ierr = PetscPOpen(comm,machine,buffer,"r",&fp);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ;
  }

  PetscFunctionReturn(0);
}

