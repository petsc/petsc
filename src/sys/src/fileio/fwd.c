#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fwd.c,v 1.16 1998/04/27 19:48:45 curfman Exp bsmith $";
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
#define __FUNC__ "PetscGetWorkingDirectory"
/*@C
   PetscGetWorkingDirectory - Gets the current working directory.

   Not Collective

   Input Parameters:
+  path - use to hold the result value
-  len  - maximum length of path

.keywords, system, get, current, working, directory
@*/
int PetscGetWorkingDirectory( char *path,int len )
{
  PetscFunctionBegin;
#if defined(HAVE_GETWD)
  getwd( path );
#elif defined(PARCH_nt)
  _getcwd( path, len );
#else
  getcwd( path, len );
#endif
  PetscFunctionReturn(0);
}
