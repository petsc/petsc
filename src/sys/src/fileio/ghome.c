#ifndef lint
static char vcid[] = "$Id: ghome.c,v 1.2 1996/02/08 18:26:06 bsmith Exp bsmith $";
#endif
/*
      Code for manipulating files.
*/
#include "files.h"

/*@C
   PetscGetHomeDirectory - Returns user's home directory name.

   Input Parameter:
.  maxlen - maximum lengh allowed

   Output Parameter:
.  dir - the home directory

.keywords: system, get, real, path

.seealso: PetscRemoveHomeDirectory()
@*/
int PetscGetHomeDirectory(int maxlen,char *dir)
{
  struct passwd *pw = 0;
  pw = getpwuid( getuid() );
  if (!pw)  return 0;
  PetscStrncpy(dir, pw->pw_dir,maxlen);
  return 0;
}
