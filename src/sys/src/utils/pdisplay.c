#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pdisplay.c,v 1.1 1998/05/18 19:27:39 bsmith Exp bsmith $";
#endif

#include "petsc.h"        
#include "sys.h"             /*I    "sys.h"   I*/
#include <stdarg.h>
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "pinclude/petscfix.h"

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
int PetscGetDisplay(char *display,int n)
{
  PetscFunctionBegin;
  PetscStrncpy(display,PetscDisplay,n);
  PetscFunctionReturn(0);  
}
