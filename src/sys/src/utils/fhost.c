#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fhost.c,v 1.30 1999/03/17 23:21:54 bsmith Exp bsmith $";
#endif
/*
      Code for manipulating files.
*/
#include "petsc.h"
#include "sys.h"
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if !defined(PARCH_win32)
#include <sys/utsname.h>
#endif
#if defined(PARCH_win32)
#include <windows.h>
#include <io.h>
#include <direct.h>
#endif
#if defined (PARCH_win32_gnu)
#include <windows.h>
#endif
#if defined(HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif
#if defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif
#include "pinclude/petscfix.h"

#undef __FUNC__  
#define __FUNC__ "PetscGetHostName"
/*@C
    PetscGetHostName - Returns the name of the host. This attempts to
    return the entire Internet name. It may not return the same name
    as MPI_Get_processor_name().

    Not Collective

    Input Parameter:
.   nlen - length of name

    Output Parameter:
.   name - contains host name.  Must be long enough to hold the name
           This is the fully qualified name, including the domain.

    Level: developer

.keywords: system, get, host, name

.seealso: PetscGetUserName()
@*/
int PetscGetHostName( char name[], int nlen )
{
#if defined(PARCH_win32) || defined(PARCH_win32_gnu)
  PetscFunctionBegin;
  GetComputerName((LPTSTR)name,(LPDWORD)(&nlen));
#elif defined(HAVE_UNAME)
  struct utsname utname;
  int            ierr;

  PetscFunctionBegin;
  uname(&utname); 
  ierr = PetscStrncpy(name,utname.nodename,nlen);CHKERRQ(ierr);
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
    /* change domain name if it is an ANL crap one */
    if (!PetscStrcmp(name+l,"qazwsxedc")) {
      int ierr = PetscStrncpy(name+l,"mcs.anl.gov",nlen-12);CHKERRQ(ierr);
    }
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
