#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ghome.c,v 1.21 1998/05/18 20:15:47 bsmith Exp balay $";
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
#if defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if !defined(PARCH_nt)
#include <sys/utsname.h>
#endif
#if defined(PARCH_nt)
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

#undef __FUNC__  
#define __FUNC__ "PetscGetHomeDirectory"
/*@C
   PetscGetHomeDirectory - Returns user's home directory name.

   Not Collective

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
  if (d1 == NULL) d1 ="c:";
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

#undef __FUNC__  
#define __FUNC__ "PetscFixFilename"
/*@C
    PetscFixFilename - Fixes a file name so that it is correct for both Unix and 
    Windows by using the correct / or \ to seperate directories.

   Not Collective

   Input Parameter:
.  name - name of file (must be in writable memory)

   Notes:
   Call PetscFixFilename() just before calling fopen().
@*/
int PetscFixFilename(char *file)
{
  int i,n;

  PetscFunctionBegin;
  if (!file) PetscFunctionReturn(0);

  n = PetscStrlen(file);
  for (i=0; i<n; i++ ) {
#if defined(PARCH_nt)
    if (file[i] == '/') file[i] = '\\';
#else
    if (file[i] == '\\') file[i] = '/';
#endif
  }
  PetscFunctionReturn(0);
}
