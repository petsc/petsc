#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fuser.c,v 1.15 1998/05/08 16:12:54 bsmith Exp bsmith $";
#endif
/*
      Code for manipulating files.
*/
#include "petsc.h"
#include "sys.h"
#if defined(HAVE_PWD_H)
#include <pwd.h>
#endif
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if !defined(PARCH_nt)
#include <sys/param.h>
#include <sys/utsname.h>
#include <windows.h>
#include <io.h>
#include <direct.h>
#endif
#if defined (PARCH_nt_gnu)
#include <windows.h>
#endif
#if defined(HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif
#include "pinclude/petscfix.h"


#if defined(HAVE_GET_USER_NAME)
#undef __FUNC__  
#define __FUNC__ "PetscGetUserName"
int PetscGetUserName( char *name, int nlen )
{
  PetscFunctionBegin;
  GetUserName((LPTSTR)name,(LPDWORD)(&nlen));
  PetscFunctionReturn(0);
}

#elif defined(HAVE_PWD_H)
#undef __FUNC__  
#define __FUNC__ "PetscGetUserName"
/*@C
    PetscGetUserName - Returns the name of the user.

    Not Collective

    Input Parameter:
    nlen - length of name

    Output Parameter:
.   name - contains user name.  Must be long enough to hold the name

.keywords: system, get, user, name

.seealso: PetscGetHostName()
@*/
int PetscGetUserName( char *name, int nlen )
{
  struct passwd *pw;

  PetscFunctionBegin;
  pw = getpwuid( getuid() );
  if (!pw) PetscStrncpy( name, "Unknown",nlen );
  else     PetscStrncpy( name, pw->pw_name,nlen );
  PetscFunctionReturn(0);
}

#else

#undef __FUNC__  
#define __FUNC__ "PetscGetUserName"
int PetscGetUserName( char *name, int nlen )
{
  PetscFunctionBegin;
  PetscStrncpy( name, "Unknown", nlen );
  PetscFunctionReturn(0);
}
#endif /* !HAVE_PWD_H */

