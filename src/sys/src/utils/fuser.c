#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fuser.c,v 1.13 1998/04/13 17:30:26 bsmith Exp curfman $";
#endif
/*
      Code for manipulating files.
*/
#include "src/sys/src/files.h"

#if defined (PARCH_nt) || defined(PARCH_nt_gnu)
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

