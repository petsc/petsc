#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: arch.c,v 1.27 1998/04/13 17:30:26 bsmith Exp curfman $";
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
           10 characters long.

.keywords: architecture, machine     
@*/
int PetscGetArchType(char *str,int slen)
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
#elif defined(PARCH_nt)
  PetscStrncpy(str,"nt",slen);
#elif defined(PARCH_nt_gnu)
  PetscStrncpy(str,"nt_gnu",slen);
#else
  PetscStrncpy(str,"Unknown",slen);
#endif
  PetscFunctionReturn(0);
}

