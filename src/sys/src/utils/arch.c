#ifndef lint
static char vcid[] = "$Id: arch.c,v 1.14 1996/04/09 17:54:50 bsmith Exp bsmith $";
#endif
#include "petsc.h"         /*I  "petsc.h"  I*/
#include "sys.h"           /*I  "sys.h"  I*/

/*@C
     PetscGetArchType - Returns a standardized architecture type for the machine
     that is executing this routine. 

     Input Parameter:
.    slen - length of string buffer

     Output Parameter:
.    str - string area to contain architecture name, should be at least 
           10 characters long.

.keywords: architecture, machine     
@*/
int PetscGetArchType(char *str,int slen)
{
#if defined(PARCH_solaris)
  PetscStrncpy(str,"solaris",slen);
#elif defined(PARCH_sun4) 
  PetscStrncpy(str,"sun4",slen);
#elif defined(PARCH_IRIX64)
  PetscStrncpy(str,"IRIX64",slen);
#elif defined(PARCH_IRIX)
  PetscStrncpy(str,"IRIX",slen);
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
#else
  PetscStrncpy(str,"Unknown",slen);
#endif
  return 0;
}

