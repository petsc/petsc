ifndef lint
static char vcid[] = "$Id: file.c,v 1.27 1995/12/31 17:17:34 curfman Exp bsmith $";
#endif
/*
      Code for manipulating files.
*/
#include "file.h"

#if defined(HAVE_PWD_H)
/*@C
    SYGetUserName - Returns the name of the user.

    Input Parameter:
    nlen - length of name

    Output Parameter:
.   name - contains user name.  Must be long enough to hold the name

.keywords: system, get, user, name

.seealso: SYGetHostName()
@*/
int SYGetUserName( char *name, int nlen )
{
  struct passwd *pw;

  pw = getpwuid( getuid() );
  if (!pw) PetscStrncpy( name, "Unknown",nlen );
  else     PetscStrncpy( name, pw->pw_name,nlen );
  return 0;
}
#else
int SYGetUserName( char *name, int nlen )
{
  PetscStrncpy( name, "Unknown", nlen );
  return 0;
}
#endif /* !HAVE_PWD_H */

