#ifndef lint
static char vcid[] = "$Id: arch.c,v 1.9 1995/09/04 17:24:00 bsmith Exp bsmith $";
#endif
#include "petsc.h"         /*I  "petsc.h"  I*/
#include "sys.h"           /*I  "sys.h"  I*/

/*@
     SYGetArchType - Return a standardized architecture type for the machine
     that is executing this routine.  This uses uname where possible,
     but may modify the name (for example, sun4 is returned for all
     sun4 types).

     Input Parameter:
     slen - length of string buffer
     Output Parameter:
.    str - string area to contain architecture name.  Should be at least 
           10 characters long.
  @*/
int SYGetArchType( char *str, int slen )
{
#if defined(PARCH_solaris)
  PetscStrncpy(str,"solaris",slen);
#elif defined(PARCH_sun4) 
  PetscStrncpy(str,"sun4",slen);
#elif defined(PARCH_IRIX)
  PetscStrncpy(str,"IRIX",slen);
#elif defined(PARCH_hpux)
  PetscStrncpy( str, "hpux", slen );
#elif defined(PARCH_rs6000)
  PetscStrncpy( str, "rs6000", slen );
#elif defined(PARCH_paragon)
  PetscStrncpy( str, "paragon", slen );
#elif defined(PARCH_t3d)
  PetscStrncpy( str, "t3d", slen );
#elif defined(PARCH_alpha)
  PetscStrncpy( str, "alpha", slen );
#else
  PetscStrncpy( str, "Unknown", slen );
#endif
  return 0;
}

