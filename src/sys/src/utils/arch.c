/*$Id: arch.c,v 1.37 2000/04/09 04:34:47 bsmith Exp bsmith $*/
#include "petsc.h"         /*I  "petsc.h"  I*/
#include "sys.h"           /*I  "sys.h"  I*/

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
#if defined(PETSC_ARCH_NAME)
  ierr = PetscStrncpy(str,PETSC_ARCH_NAME,slen);CHKERRQ(ierr);
#else
#error "bmake/$PETSC_ARCH/petscconf.h is missing PETSC_ARCH_NAME"
#endif
  PetscFunctionReturn(0);
}

