/*$Id: arch.c,v 1.44 2001/03/23 23:20:45 balay Exp $*/
#include "petsc.h"         /*I  "petsc.h"  I*/
#include "petscsys.h"           /*I  "petscsys.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscGetArchType"
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

     Concepts: machine type
     Concepts: architecture

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

