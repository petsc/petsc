#ifndef lint
static char vcid[] = "$Id: ghome.c,v 1.7 1997/02/22 02:23:29 bsmith Exp balay $";
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

   Note:
   On Windows NT machine the enviornmental variable HOME specifies the home directory.

.keywords: system, get, real, path

.seealso: PetscRemoveHomeDirectory()
@*/
int PetscGetHomeDirectory(int maxlen,char *dir)
{
  /* On NT get the HOME DIR from the env variable HOME */
#if defined(PARCH_nt) || defined(PARCH_nt_gnu)
  char *d1 = getenv("HOME");
  PetscStrncpy(dir,d1,maxlen);
#else
  struct passwd *pw = 0;
  pw = getpwuid( getuid() );
  if (!pw)  return 0;
  PetscStrncpy(dir, pw->pw_dir,maxlen);
#endif
  return 0;
}

