#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ghome.c,v 1.12 1997/08/22 15:11:48 bsmith Exp bsmith $";
#endif
/*
      Code for manipulating files.
*/
#include "src/sys/src/files.h"

#undef __FUNC__  
#define __FUNC__ "PetscGetHomeDirectory"
/*@C
   PetscGetHomeDirectory - Returns user's home directory name.

   Input Parameter:
.  maxlen - maximum lengh allowed

   Output Parameter:
.  dir - the home directory

   Note:
   On Windows NT machine the enviornmental variable HOME specifies the home directory.

.keywords: system, get, real, path

@*/
int PetscGetHomeDirectory(int maxlen,char *dir)
{
#if defined(PARCH_nt) || defined(PARCH_nt_gnu)
  char *d1 = getenv("HOME");

  PetscFunctionBegin;
  if (d1 == NULL) d1 ="c:\\";
  PetscStrncpy(dir,d1,maxlen);
#else
  struct passwd *pw = 0;

  PetscFunctionBegin;
  pw = getpwuid( getuid() );
  if (!pw)  PetscFunctionReturn(0);
  PetscStrncpy(dir, pw->pw_dir,maxlen);
#endif
  PetscFunctionReturn(0);
}

