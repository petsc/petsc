#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: arch.c,v 1.33 1999/03/17 23:21:54 bsmith Exp bsmith $";
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
  int ierr;

  PetscFunctionBegin;
#if defined(PETSC_ARCH_NAME)
  ierr = PetscStrncpy(str,PETSC_ARCH_NAME,slen);CHKERRQ(ierr);
#elif defined(PARCH_solaris)
  ierr = PetscStrncpy(str,"solaris",slen);CHKERRQ(ierr);
#elif defined(PARCH_sun4) 
  ierr = PetscStrncpy(str,"sun4",slen);CHKERRQ(ierr);
#elif defined(PARCH_IRIX64)
  ierr = PetscStrncpy(str,"IRIX64",slen);CHKERRQ(ierr);
#elif defined(PARCH_IRIX)
  ierr = PetscStrncpy(str,"IRIX",slen);CHKERRQ(ierr);
#elif defined(PARCH_IRIX5)
  ierr = PetscStrncpy(str,"IRIX5",slen);CHKERRQ(ierr);
#elif defined(PARCH_hpux)
  ierr = PetscStrncpy(str,"hpux",slen);CHKERRQ(ierr);
#elif defined(PARCH_rs6000)
  ierr = PetscStrncpy(str,"rs6000",slen);CHKERRQ(ierr);
#elif defined(PARCH_paragon)
  ierr = PetscStrncpy(str,"paragon",slen);CHKERRQ(ierr);
#elif defined(PARCH_t3d)
  ierr = PetscStrncpy(str,"t3d",slen);CHKERRQ(ierr);
#elif defined(PARCH_alpha)
  ierr = PetscStrncpy(str,"alpha",slen);CHKERRQ(ierr);
#elif defined(PARCH_freebsd)
  ierr = PetscStrncpy(str,"freebsd",slen);CHKERRQ(ierr);
#elif defined(PARCH_win32)
  ierr = PetscStrncpy(str,"win32",slen);CHKERRQ(ierr);
#elif defined(PARCH_win32_gnu)
  ierr = PetscStrncpy(str,"win32_gnu",slen);CHKERRQ(ierr);
#elif defined(PARCH_linux)
  ierr = PetscStrncpy(str,"linux",slen);CHKERRQ(ierr);
#else
  ierr = PetscStrncpy(str,"Unknown",slen);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

