#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: arch.c,v 1.32 1998/12/17 21:56:12 balay Exp bsmith $";
#endif
#include "petsc.h"         /*I  "petsc.h"  I*/
#include "sys.h"           /*I  "sys.h"  I*/

#undef __FUNC__  
#define __FUNC__ "PetscGetArchType"
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
  PetscFunctionBegin;
#if defined(PETSC_ARCH_NAME)
  PetscStrncpy(str,PETSC_ARCH_NAME,slen);
#elif defined(PARCH_solaris)
  PetscStrncpy(str,"solaris",slen);
#elif defined(PARCH_sun4) 
  PetscStrncpy(str,"sun4",slen);
#elif defined(PARCH_IRIX64)
  PetscStrncpy(str,"IRIX64",slen);
#elif defined(PARCH_IRIX)
  PetscStrncpy(str,"IRIX",slen);
#elif defined(PARCH_IRIX5)
  PetscStrncpy(str,"IRIX5",slen);
#elif defined(PARCH_hpux)
  PetscStrncpy(str,"hpux",slen);
#elif defined(PARCH_rs6000)
  PetscStrncpy(str,"rs6000",slen);
#elif defined(PARCH_paragon)
  PetscStrncpy(str,"paragon",slen);
#elif defined(PARCH_t3d)
  PetscStrncpy(str,"t3d",slen);
#elif defined(PARCH_alpha)
  PetscStrncpy(str,"alpha",slen);
#elif defined(PARCH_freebsd)
  PetscStrncpy(str,"freebsd",slen);
#elif defined(PARCH_win32)
  PetscStrncpy(str,"win32",slen);
#elif defined(PARCH_win32_gnu)
  PetscStrncpy(str,"win32_gnu",slen);
#elif defined(PARCH_linux)
  PetscStrncpy(str,"linux",slen);
#else
  PetscStrncpy(str,"Unknown",slen);
#endif
  PetscFunctionReturn(0);
}

