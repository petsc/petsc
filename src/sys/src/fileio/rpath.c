#ifndef lint
static char vcid[] = "$Id: file.c,v 1.27 1995/12/31 17:17:34 curfman Exp bsmith $";
#endif
/*
      Code for manipulating files.
*/
#include "file.h"


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

  /* Find last '/' */
  p = PetscStrrchr( fullpath, '/' );
  if (!p) p = fullpath;
  else    p++;
  PetscStrncpy( path, p, flen );
  return 0;
}
