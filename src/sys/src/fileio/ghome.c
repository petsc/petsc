#ifndef lint
static char vcid[] = "$Id: ghome.c,v 1.4 1996/08/08 14:41:26 bsmith Exp balay $";
#endif
/*
      Code for manipulating files.
*/
#include "src/sys/src/files.h"

#undef __FUNCTION__  
#define __FUNCTION__ "PetscGetHomeDirectory"
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
