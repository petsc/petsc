#ifndef lint
static char vcid[] = "$Id: fuser.c,v 1.6 1997/01/06 20:22:55 balay Exp bsmith $";
#endif
/*
      Code for manipulating files.
*/
#include "src/sys/src/files.h"

#if defined(HAVE_PWD_H)
#undef __FUNC__  
#define __FUNC__ "PetscGetUserName" /* ADIC Ignore */
/*@C
    PetscGetUserName - Returns the name of the user.

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

  pw = getpwuid( getuid() );
  if (!pw) PetscStrncpy( name, "Unknown",nlen );
  else     PetscStrncpy( name, pw->pw_name,nlen );
  return 0;
}
#else
#undef __FUNC__  
#define __FUNC__ "PetscGetUserName" /* ADIC Ignore */
int PetscGetUserName( char *name, int nlen )
{
  PetscStrncpy( name, "Unknown", nlen );
  return 0;
}
#endif /* !HAVE_PWD_H */

