#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: rpath.c,v 1.13 1998/04/27 19:48:45 curfman Exp bsmith $";
#endif
/*
      Code for manipulating files.
*/
#include "petsc.h"
#include "sys.h"
#include "pinclude/ptime.h"
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
#include <sys/param.h>
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
#include <fcntl.h>
#include <time.h>  
#if defined(HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif
#include "pinclude/petscfix.h"

#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif


#undef __FUNC__  
#define __FUNC__ "PetscGetRelativePath"
/*@C
   PetscGetRelativePath - Given a filename, returns the relative path (removes
   all directory specifiers).

   Not Collective

   Input parameters:
+  fullpath  - full pathname
.  path      - pointer to buffer to hold relative pathname
-  flen     - size of path

.keywords: system, get, relative, path

.seealso: PetscGetFullPath()
@*/
int PetscGetRelativePath( char *fullpath, char *path, int flen )
{
  char  *p;

  PetscFunctionBegin;
  /* Find string after last '/' or entire string if no '/' */
  p = PetscStrrchr( fullpath, '/' );
  PetscStrncpy( path, p, flen );
  PetscFunctionReturn(0);
}
