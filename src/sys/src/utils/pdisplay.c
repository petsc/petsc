/*$Id: pdisplay.c,v 1.12 1999/11/05 14:44:20 bsmith Exp bsmith $*/

#include "petsc.h"        
#include "sys.h"             /*I    "sys.h"   I*/
#include <stdarg.h>
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "petscfix.h"

#undef __FUNC__  
#define __FUNC__ "OptionsGetenv"
/*@C
     OptionsGetenv - Gets an environmental variable, broadcasts to all
          processors in communicator from first.

     Collective on MPI_Comm

   Input Parameters:
+    comm - communicator to share variable
.    name - name of environmental variable
-    len - amount of space allocated to hold variable

   Output Parameters:
+    flag - if not PETSC_NULL tells if variable found or not
-    env - value of variable

  Level: advanced

   Notes:
    If comm does not contain the 0th process in the MPIRUN it is likely on
    many systems that the environmental variable will not be set unless you
    put it in a universal location like a .chsrc file

@*/
int OptionsGetenv(MPI_Comm comm,const char *name,char env[],int len,PetscTruth *flag)
{
  int        rank,ierr;
  char       *str;
  PetscTruth flg = PETSC_FALSE;
   
  PetscFunctionBegin;
  ierr = PetscMemzero(env,len*sizeof(char));CHKERRQ(ierr);

  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    str = getenv(name);
    if (str) flg = PETSC_TRUE;
    if (str && env) {ierr = PetscStrncpy(env,str,len);CHKERRQ(ierr);}
  }
  ierr = MPI_Bcast(&flg,1,MPI_INT,0,comm);CHKERRQ(ierr);
  ierr = MPI_Bcast(env,len,MPI_CHAR,0,comm);CHKERRQ(ierr);
  if (flag) {
    *flag = flg;
  } 
  PetscFunctionReturn(0);
}

/*
     PetscSetDisplay - Tries to set the X windows display variable for all processors.
                       The variable PetscDisplay contains the X windows display variable.

*/
static char PetscDisplay[128]; 

#undef __FUNC__  
#define __FUNC__ "PetscSetDisplay" 
int PetscSetDisplay(void)
{
  int        size,rank,len,ierr;
  PetscTruth flag;
  char       *str;

  PetscFunctionBegin;
  ierr = OptionsGetString(PETSC_NULL,"-display",PetscDisplay,128,&flag);CHKERRQ(ierr);
  if (flag) PetscFunctionReturn(0);

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr); 
  if (!rank) {
    str = getenv("DISPLAY");
    if (!str || (str[0] == ':' && size > 1)) {
      ierr = PetscGetHostName(PetscDisplay,124);CHKERRQ(ierr);
      ierr = PetscStrcat(PetscDisplay,":0.0");CHKERRQ(ierr);
    } else {
      ierr = PetscStrncpy(PetscDisplay,str,128);CHKERRQ(ierr);
    }
    ierr = PetscStrlen(PetscDisplay,&len);CHKERRQ(ierr);
    ierr = MPI_Bcast(&len,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = MPI_Bcast(PetscDisplay,len,MPI_CHAR,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  } else {
    ierr = MPI_Bcast(&len,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = MPI_Bcast(PetscDisplay,len,MPI_CHAR,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
    PetscDisplay[len] = 0;
  }
  PetscFunctionReturn(0);  
}

#undef __FUNC__  
#define __FUNC__ "PetscGetDisplay" 
/*
     PetscGetDisplay - Gets the display variable for all processors.

  Input Parameters:
.   n - length of string display

  Output Parameters:
.   display - the display string

*/
int PetscGetDisplay(char display[],int n)
{
  PetscFunctionBegin;
  PetscStrncpy(display,PetscDisplay,n);
  PetscFunctionReturn(0);  
}
