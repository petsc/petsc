/*$Id: rpath.c,v 1.31 2001/03/23 23:20:30 balay Exp $*/

#include "petsc.h"
#include "petscsys.h"
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
#if !defined(PARCH_win32)
#include <sys/utsname.h>
#endif
#if defined(PARCH_win32)
#include <windows.h>
#include <io.h>
#include <direct.h>
#endif
#if defined (PARCH_win32_gnu)
#include <windows.h>
#endif
#if defined(HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif
#include "petscfix.h"

#undef __FUNCT__  
#define __FUNCT__ "PetscGetRelativePath"
/*@C
   PetscGetRelativePath - Given a filename, returns the relative path (removes
   all directory specifiers).

   Not Collective

   Input parameters:
+  fullpath  - full pathname
.  path      - pointer to buffer to hold relative pathname
-  flen     - size of path

   Level: developer

   Concepts: relative path
   Concepts: path^relative

.seealso: PetscGetFullPath()
@*/
int PetscGetRelativePath(const char fullpath[],char path[],int flen)
{
  char  *p;
  int   ierr;

  PetscFunctionBegin;
  /* Find string after last / or entire string if no / */
  ierr = PetscStrrchr(fullpath,'/',&p);CHKERRQ(ierr);
  ierr = PetscStrncpy(path,p,flen);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
