#ifndef lint
static char vcid[] = "$Id: fhost.c,v 1.3 1996/03/19 21:24:22 bsmith Exp bsmith $";
#endif
/*
      Code for manipulating files.
*/
#include "files.h"

/*@C
    PetscGetHostName - Returns the name of the host.

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
  struct utsname utname;
  /* Note we do not use gethostname since that is not POSIX */
  uname(&utname); PetscStrncpy(name,utname.nodename,nlen);

  /* See if this name includes the domain */
  if (!PetscStrchr(name,'.')) {
    int  l;
    l = PetscStrlen(name);
    if (l == nlen) return 0;
    name[l++] = '.';
#if defined(PARCH_solaris)
    sysinfo( SI_SRPC_DOMAIN,name+l,nlen-l);
#elif defined(HAVE_GETDOMAINNAME)
    getdomainname( name+l, nlen - l );
#endif
  }
  return 0;
}
