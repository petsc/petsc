#ifndef lint
static char vcid[] = "$Id: arch.c,v 1.1 1994/10/04 19:58:38 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include <string.h>


/*
    Note:
    On a CM5, the system call that this routine uses (uname) is provided but 
    causes a fatal error.  
 */
#define HAS_UNAME
#if defined(MSDOS) || defined(intelnx) || defined(fx2800) || defined(cm5) || \
    defined(NeXT)
#undef HAS_UNAME
#endif
#if defined(dec5000) || defined(rs6000)
#undef HAS_UNAME
#endif
#if defined(sun4) && defined(sun4Pre41)
#undef HAS_UNAME
#endif

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
void SYGetArchType( str, slen )
char *str;
int  slen;
{
#if defined(HAS_UNAME)
#include <sys/utsname.h>
struct utsname un;
uname( &un );

strncpy( str, un.machine, slen );

/* Here is special code for each variety */
#if defined(solaris)
strncpy(str,"solaris",7);
#elif defined(sun4) 
if (strncmp( str, "sun4", 4 ) == 0) 
    str[4] = 0;   /* Remove any trailing version, such as "sun4c" */
else if (strncmp( str, "tp_s1", 5 ) == 0)
    strcpy( str, "sun4" );   /* Tadpole (SPARC notebook) */

#elif defined(IRIX)
if (strcmp( "IRIX", un.sysname ) != 0)
    strcpy( str, "Unknown/IRIX" );  
else
    strcpy( str, "IRIX" );

#elif defined(tc2000)
strcpy( str, "tc2000" );   /* Need a lowercase version eventually ... */
#endif

#elif defined(HPUX)
strncpy( str, "hpux", slen );

#elif defined(fx2800)
strncpy( str, "fx2800", slen );

#elif defined(rs6000)
strncpy( str, "rs6000", slen );

#elif defined(MSDOS)
strncpy( str, "msdos", slen );

#elif defined(intelnx)
strncpy( str, "intelnx", slen );

#elif defined(sun4)
/* For Sparc systems before uname (4.0.3c and earlier) */
strncpy( str, "sun4", slen );

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
