#ifndef lint
static char vcid[] = "$Id: fuser.c,v 1.3 1996/03/19 21:24:22 bsmith Exp bsmith $";
#endif
/*
      Code for manipulating files.
*/
#include "src/sys/src/files.h"

#if defined(HAVE_PWD_H)
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
int PetscGetUserName( char *name, int nlen )
{
  PetscStrncpy( name, "Unknown", nlen );
  return 0;
}
#endif /* !HAVE_PWD_H */

