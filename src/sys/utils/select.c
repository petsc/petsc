
#include <petscsys.h>              /*I  "petscsys.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscPopUpSelect"
/*@C
     PetscPopUpSelect - Pops up a windows with a list of choices; allows one to be chosen

     Collective on MPI_Comm

     Input Parameters:
+    comm - MPI communicator, all processors in communicator must call this but input 
            from first communicator is the only one that is used
.    machine - location to run popup program or PETSC_NULL
.    title - text to display above choices
.    n - number of choices
-    choices - array of strings

     Output Parameter:
.    choice - integer indicating which one was selected

     Level: developer

     Notes:
       Uses DISPLAY variable or -display option to determine where it opens the window

       Currently this uses a file ~username/.popuptmp to pass the value back from the 
       xterm; hence this program must share a common file system with the machine
       parameter passed in below.

   Concepts: popup
   Concepts: user selection
   Concepts: menu

@*/
PetscErrorCode  PetscPopUpSelect(MPI_Comm comm,const char *machine,const char *title,int n,const char **choices,int *choice)
{
  PetscMPIInt    rank;
  int            i,rows = n + 2;
  size_t         cols,len;
  char           buffer[2048],display[256],geometry[64];
  FILE           *fp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!title) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Must pass in a title line");
  if (n < 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must pass in at least one selection");
  if (n == 1) {*choice = 0; PetscFunctionReturn(0);}

  ierr = PetscStrlen(title,&cols);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscStrlen(choices[i],&len);CHKERRQ(ierr);
    cols = PetscMax(cols,len);
  }
  cols += 4;
  sprintf(geometry," -geometry %dx%d ",(int)cols,rows);
  ierr = PetscStrcpy(buffer,"xterm -bw 100 -bd blue +sb -display ");CHKERRQ(ierr);
  ierr = PetscGetDisplay(display,128);CHKERRQ(ierr);
  ierr = PetscStrcat(buffer,display);CHKERRQ(ierr);
  ierr = PetscStrcat(buffer,geometry);CHKERRQ(ierr);
  ierr = PetscStrcat(buffer," -e ${PETSC_DIR}/bin/popup ");CHKERRQ(ierr);

  ierr = PetscStrcat(buffer,"\"");CHKERRQ(ierr);
  ierr = PetscStrcat(buffer,title);CHKERRQ(ierr);
  ierr = PetscStrcat(buffer,"\" ");CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscStrcat(buffer,"\"");CHKERRQ(ierr);
    ierr = PetscStrcat(buffer,choices[i]);CHKERRQ(ierr);
    ierr = PetscStrcat(buffer,"\" ");CHKERRQ(ierr);
  }
#if defined(PETSC_HAVE_POPEN)
  ierr = PetscPOpen(comm,machine,buffer,"r",&fp);CHKERRQ(ierr);
  ierr = PetscPClose(comm,fp);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    FILE *fd;

    ierr = PetscFOpen(PETSC_COMM_SELF,"${HOMEDIRECTORY}/.popuptmp","r",&fd);CHKERRQ(ierr);
    if (fscanf(fd,"%d",choice) != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fscanf() could not read numeric choice");
    *choice -= 1;
    if (*choice < 0 || *choice > n-1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Selection %d out of range",*choice);
    ierr = PetscFClose(PETSC_COMM_SELF,fd);CHKERRQ(ierr);
  }
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Cannot run external programs on this machine");
#endif
  ierr = MPI_Bcast(choice,1,MPI_INT,0,comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

