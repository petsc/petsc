#ifndef lint
static char vcid[] = "$Id: rpath.c,v 1.2 1996/02/08 18:26:06 bsmith Exp bsmith $";
#endif
/*
      Code for manipulating files.
*/
#include "files.h"


/*@C
   SYGetRelativePath - Given a filename, returns the relative path (removes
   all directory specifiers).

   Input parameters:
.  fullpath  - full pathname
.  path      - pointer to buffer to hold relative pathname
.  flen     - size of path

.keywords: system, get, relative, path

.seealso: SYGetFullPath()
@*/
int SYGetRelativePath( char *fullpath, char *path, int flen )
{
  char  *p;

  /* Find string after last '/' or entire string if no '/' */
  p = PetscStrrchr( fullpath, '/' );
  PetscStrncpy( path, p, flen );
  return 0;
}
