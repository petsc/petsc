#ifndef lint
static char vcid[] = "$Id: rpath.c,v 1.7 1997/01/06 20:22:55 balay Exp bsmith $";
#endif
/*
      Code for manipulating files.
*/
#include "src/sys/src/files.h"


#undef __FUNC__  
#define __FUNC__ "PetscGetRelativePath" /* ADIC Ignore */
/*@C
   PetscGetRelativePath - Given a filename, returns the relative path (removes
   all directory specifiers).

   Input parameters:
.  fullpath  - full pathname
.  path      - pointer to buffer to hold relative pathname
.  flen     - size of path

.keywords: system, get, relative, path

.seealso: PetscGetFullPath()
@*/
int PetscGetRelativePath( char *fullpath, char *path, int flen )
{
  char  *p;

  /* Find string after last '/' or entire string if no '/' */
  p = PetscStrrchr( fullpath, '/' );
  PetscStrncpy( path, p, flen );
  return 0;
}
