/*$Id: arch.c,v 1.39 2000/05/05 22:14:11 balay Exp balay $*/
#include "petsc.h"         /*I  "petsc.h"  I*/
#include "petscsys.h"           /*I  "petscsys.h"  I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscGetArchType"
/*@C
     PetscGetArchType - Returns a standardized architecture type for the machine
     that is executing this routine. 

     Not Collective

     Input Parameter:
.    slen - length of string buffer

     Output Parameter:
.    str - string area to contain architecture name, should be at least 
           10 characters long. Name is truncated if string is not long enough.

     Level: developer

.keywords: architecture, machine     
@*/
int PetscGetArchType(char str[],int slen)
{
  int ierr;

  PetscFunctionBegin;
  ierr = PetscStrncpy(str,PETSC_ARCH,slen);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

