
#include "petsc.h"
#include <string.h>


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
void SYGetArchType( char *str, int slen )
{
#if defined(PARCH_solaris)
strncpy(str,"solaris",7);
#elif defined(PARCH_sun4) 
strncpy(str,"sun4",4);
#elif defined(PARCH_IRIX)
strncpy(str,"IRIX",4);
#elif defined(PARCH_tc2000)
strcpy( str, "tc2000" );   /* Need a lowercase version eventually ... */
#elif defined(PARCH_HPUX)
strncpy( str, "hpux", 4 );
#elif defined(PARCH_fx2800)
strncpy( str, "fx2800", 6 );
#elif defined(PARCH_rs6000)
strncpy( str, "rs6000", slen );
#elif defined(PARCH_MSDOS)
strncpy( str, "msdos", slen );
#elif defined(PARCH_intelnx)
strncpy( str, "intelnx", slen );
#elif defined(PARCH_dec5000)
strncpy( str, "dec5000", slen );
#elif defined(PARCH_tc2000)
strncpy( str, "tc2000", slen );
#elif defined(PARCH_cm5)
strncpy( str, "cm5", slen );
#elif defined(PARCH_NeXT)
strncpy( str, "NeXT", slen );
#else
strncpy( str, "Unknown", slen );
#endif
}

