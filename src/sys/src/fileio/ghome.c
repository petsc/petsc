#ifndef lint
static char vcid[] = "$Id: ghome.c,v 1.6 1997/01/06 20:22:55 balay Exp bsmith $";
#endif
/*
      Code for manipulating files.
*/
#include "src/sys/src/files.h"

#undef __FUNC__  
#define __FUNC__ "PetscGetHomeDirectory" /* ADIC Ignore */
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
