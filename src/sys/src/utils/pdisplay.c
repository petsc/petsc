#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pdisplay.c,v 1.3 1998/08/26 22:01:52 balay Exp bsmith $";
#endif

#include "petsc.h"        
#include "sys.h"             /*I    "sys.h"   I*/
#include <stdarg.h>
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "pinclude/petscfix.h"

#undef __FUNC__  
#define __FUNC__ "OptionsGetenv" /*
     OptionsGetenv - Gets an environmental variable, broadcasts to all
          processors in communicator from first.

    comm - communicator to share variable

    name - name of environmental variable
    len - amount of space allocated to hold variable
    flag - if not PETSC_NULL tells if variable found or not

*/
int OptionsGetenv(MPI_Comm comm,const char *name,char env[],int len,int *flag)
{
  int  rank,ierr;
  char *str;
   
  PetscFunctionBegin;
  PetscMemzero(env,len*sizeof(char));

  MPI_Comm_rank(comm,&rank);
  if (!rank) {
    str = getenv(name);
    if (str) PetscStrncpy(env,str,len);
  }
  ierr = MPI_Bcast(env,len,MPI_CHAR,0,comm);CHKERRQ(ierr);
  if (flag && env[0]) {
    *flag = 1;
  } else if (flag) {
    *flag = 0;
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
  int  size,rank,len,ierr,flag;
  char *str;

  PetscFunctionBegin;
  ierr = OptionsGetString(0,"-display",PetscDisplay,128,&flag);CHKERRQ(ierr);
  if (flag) PetscFunctionReturn(0);

  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);  
  if (!rank) {
    str = getenv("DISPLAY");
    if (!str || (str[0] == ':' && size > 1)) {
      ierr = PetscGetHostName(PetscDisplay,124); CHKERRQ(ierr);
      PetscStrcat(PetscDisplay,":0.0");
    } else {
      PetscStrncpy(PetscDisplay,str,128);
    }
    len  = PetscStrlen(PetscDisplay);
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
.   display - the display string, may (and should) be freed.

*/
int PetscGetDisplay(char display[],int n)
{
  PetscFunctionBegin;
  PetscStrncpy(display,PetscDisplay,n);
  PetscFunctionReturn(0);  
}
