#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fhost.c,v 1.18 1997/08/22 15:11:48 bsmith Exp bsmith $";
#endif
/*
      Code for manipulating files.
*/
#include "src/sys/src/files.h"   /*I  "sys.h"   I*/
#undef __FUNC__  
#define __FUNC__ "PetscGetHostName"
/*@C
    PetscGetHostName - Returns the name of the host. This attempts to
    return the entire Internet name. It may not return the same name
    as MPI_Get_processor_name().

    Input Parameter:
    nlen - length of name

    Output Parameter:
.   name - contains host name.  Must be long enough to hold the name
           This is the fully qualified name, including the domain.

.keywords: system, get, host, name

.seealso: PetscGetUserName()
@*/
int PetscGetHostName( char *name, int nlen )
{
#if defined(PARCH_nt) || defined(PARCH_nt_gnu)
  PetscFunctionBegin;
  GetComputerName((LPTSTR)name,(LPDWORD)(&nlen));
#elif defined(HAVE_UNAME)
  struct utsname utname;
  PetscFunctionBegin;
  uname(&utname); 
  PetscStrncpy(name,utname.nodename,nlen);
#elif defined(HAVE_GETHOSTNAME)
  PetscFunctionBegin;
  gethostname(name, nlen);
#elif defined(HAVE_SYSINFO)
  PetscFunctionBegin;
  sysinfo(SI_HOSTNAME, name, nlen);
#endif
  /* See if this name includes the domain */
  if (!PetscStrchr(name,'.')) {
    int  l;
    l = PetscStrlen(name);
    if (l == nlen) PetscFunctionReturn(0);
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
    if (!PetscStrncmp(name+l,"(none)",6)) {
      name[l-1] = 0;
    }
  }
  PetscFunctionReturn(0);
}
