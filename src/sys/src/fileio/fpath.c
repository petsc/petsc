#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fpath.c,v 1.22 1998/12/17 21:57:17 balay Exp bsmith $";
#endif
/*
      Code for opening and closing files.
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
#include <fcntl.h>
#if defined(HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif
#include "pinclude/petscfix.h"

#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif

#if defined(HAVE_PWD_H)

#undef __FUNC__  
#define __FUNC__ "PetscGetFullPath"
/*@C
   PetscGetFullPath - Given a filename, returns the fully qualified file name.

   Not Collective

   Input Parameters:
+  path     - pathname to qualify
.  fullpath - pointer to buffer to hold full pathname
-  flen     - size of fullpath

   Level: developer

.keywords: system, get, full, path

.seealso: PetscGetRelativePath()
@*/
int PetscGetFullPath( const char path[], char fullpath[], int flen )
{
  struct passwd *pwde;
  int           ln;

  PetscFunctionBegin;
  if (path[0] == '/') {
    if (PetscStrncmp("/tmp_mnt/",path,9) == 0) PetscStrncpy(fullpath, path + 8, flen);
    else PetscStrncpy( fullpath, path, flen); 
    PetscFunctionReturn(0);
  }
  PetscGetWorkingDirectory( fullpath, flen );
  PetscStrncat( fullpath,"/",flen - PetscStrlen(fullpath) );
  if ( path[0] == '.' && path[1] == '/' ) {
    PetscStrncat( fullpath, path+2, flen - PetscStrlen(fullpath) - 1 );
  } else {
    PetscStrncat( fullpath, path, flen - PetscStrlen(fullpath) - 1 );
  }

  /* Remove the various "special" forms (~username/ and ~/) */
  if (fullpath[0] == '~') {
    char tmppath[MAXPATHLEN];
    if (fullpath[1] == '/') {
	pwde = getpwuid( geteuid() );
	if (!pwde) PetscFunctionReturn(0);
	PetscStrcpy( tmppath, pwde->pw_dir );
	ln = PetscStrlen( tmppath );
	if (tmppath[ln-1] != '/') PetscStrcat( tmppath+ln-1, "/" );
	PetscStrcat( tmppath, fullpath + 2 );
	PetscStrncpy( fullpath, tmppath, flen );
    } else {
	char *p, *name;

	/* Find username */
	name = fullpath + 1;
	p    = name;
	while (*p && isalnum((int)(*p))) p++;
	*p = 0; p++;
	pwde = getpwnam( name );
	if (!pwde) PetscFunctionReturn(0);
	
	PetscStrcpy( tmppath, pwde->pw_dir );
	ln = PetscStrlen( tmppath );
	if (tmppath[ln-1] != '/') PetscStrcat( tmppath+ln-1, "/" );
	PetscStrcat( tmppath, p );
	PetscStrncpy( fullpath, tmppath, flen );
    }
  }
  /* Remove the automounter part of the path */
  if (PetscStrncmp( fullpath, "/tmp_mnt/", 9 ) == 0) {
    char tmppath[MAXPATHLEN];
    PetscStrcpy( tmppath, fullpath + 8 );
    PetscStrcpy( fullpath, tmppath );
  }
  /* We could try to handle things like the removal of .. etc */
  PetscFunctionReturn(0);
}
#elif defined (PARCH_win32)
#undef __FUNC__  
#define __FUNC__ "PetscGetFullPath"
int PetscGetFullPath(const char path[],char fullpath[], int flen )
{
  PetscFunctionBegin;
  _fullpath(fullpath,path,flen);
  PetscFunctionReturn(0);
}
#else
#undef __FUNC__  
#define __FUNC__ "PetscGetFullPath"
int PetscGetFullPath(const char path[],char fullpath[], int flen )
{
  PetscFunctionBegin;
  PetscStrcpy( fullpath, path );
  PetscFunctionReturn(0);
}	
#endif
