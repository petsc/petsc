#ifndef lint
static char vcid[] = "$Id: fhost.c,v 1.8 1996/08/08 14:41:26 bsmith Exp gropp $";
#endif
/*
      Code for manipulating files.
*/
#include "src/sys/src/files.h"   /*I  "sys.h"   I*/

/*@C
    PetscGetHostName - Returns the name of the host. This attempts to
      return the entire internet name. It may not return the same name
      as MPI_Get_processor_name().

    Input Parameter:
    nlen - length of name

    Output Parameter:
.   name - contains host name.  Must be long enough to hold the name
           This is the fully qualified name, including the domain.

.keywords: syetem, get, host, name

.seealso: PetscGetUserName()
@*/
int PetscGetHostName( char *name, int nlen )
{
/* The "1" in this test is for eventual elimination when all of the "base"
   files specify HAVE_UNAME (if they should!) */
#if 1 || defined(HAVE_UNAME)
  struct utsname utname;
  /* Note we do not use gethostname since that is not POSIX */
  uname(&utname); PetscStrncpy(name,utname.nodename,nlen);
#elif defined(HAVE_GETHOSTNAME)
    gethostname(name, nlen);
#elif defined(HAVE_SYSINFO)
    sysinfo(SI_HOSTNAME, name, nlen);
#endif
  /* See if this name includes the domain */
  if (!PetscStrchr(name,'.')) {
    int  l;
    l = PetscStrlen(name);
    if (l == nlen) return 0;
    name[l++] = '.';
#if defined(HAVE_SYSINFO)
    sysinfo( SI_SRPC_DOMAIN,name+l,nlen-l);
#elif defined(HAVE_GETDOMAINNAME)
    getdomainname( name+l, nlen - l );
#endif
    /* 
       Some machines (Linx) default to (none) if not
       configured with a particular domain name.
    */
    if (PetscStrncmp(name+l,"(none)",6)) {
      name[l-1] = 0;
    }
  }
  return 0;
}

