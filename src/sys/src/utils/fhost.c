#ifndef lint
static char vcid[] = "$Id: fhost.c,v 1.1 1996/01/30 18:31:10 bsmith Exp bsmith $";
#endif
/*
      Code for manipulating files.
*/
#include "files.h"

/*@C
    SYGetHostName - Returns the name of the host.

    Input Parameter:
    nlen - length of name

    Output Parameter:
.   name - contains host name.  Must be long enough to hold the name
           This is the fully qualified name, including the domain.

.keywords: syetem, get, host, name

.seealso: SYGetUserName()
@*/
int SYGetHostName( char *name, int nlen )
{
#if defined(PARCH_solaris)
  struct utsname utname;
  uname(&utname);
  PetscStrncpy(name,utname.nodename,nlen);
#else
  gethostname(name, nlen);
#endif
/* See if this name includes the domain */
  if (!PetscStrchr(name,'.')) {
    int  l;
    l = PetscStrlen(name);
    name[l++] = '.';
#if defined(PARCH_solaris)
    sysinfo( SI_SRPC_DOMAIN,name+l,nlen-l);
#elif !defined(PARCH_t3d)
    getdomainname( name+l, nlen - l );
#endif
  }
  return 0;
}
