
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
#if defined(solaris)
strncpy(str,"solaris",7);
#elif defined(sun4) 
strncpy(str,"sun4",4);
#elif defined(IRIX)
strncpy(str,"IRIX",4);
#elif defined(tc2000)
strcpy( str, "tc2000" );   /* Need a lowercase version eventually ... */
#elif defined(HPUX)
strncpy( str, "hpux", 4 );
#elif defined(fx2800)
strncpy( str, "fx2800", 6 );
#elif defined(rs6000)
strncpy( str, "rs6000", slen );
#elif defined(MSDOS)
strncpy( str, "msdos", slen );
#elif defined(intelnx)
strncpy( str, "intelnx", slen );
#elif defined(dec5000)
strncpy( str, "dec5000", slen );
#elif defined(tc2000)
strncpy( str, "tc2000", slen );
#elif defined(cm5)
strncpy( str, "cm5", slen );
#elif defined(NeXT)
strncpy( str, "NeXT", slen );
#else
strncpy( str, "Unknown", slen );
#endif
}

