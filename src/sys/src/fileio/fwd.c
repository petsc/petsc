#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fwd.c,v 1.21 1998/12/17 21:57:10 balay Exp bsmith $";
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
#include "pinclude/petscfix.h"

#undef __FUNC__  
#define __FUNC__ "PetscGetWorkingDirectory"
/*@C
   PetscGetWorkingDirectory - Gets the current working directory.

   Not Collective

   Input Parameters:
.  len  - maximum length of path

   Output Parameter:
.  path - use to hold the result value. The string should be long enough
          to hold the path.

   Level: developer

.keywords, system, get, current, working, directory
@*/
int PetscGetWorkingDirectory( char path[],int len )
{
  PetscFunctionBegin;
#if defined(HAVE_GETWD)
  getwd( path );
#elif defined(PARCH_win32)
  _getcwd( path, len );
#else
  getcwd( path, len );
#endif
  PetscFunctionReturn(0);
}
