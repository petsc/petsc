#ifndef lint
static char vcid[] = "$Id: fwd.c,v 1.2 1996/02/08 18:26:06 bsmith Exp bsmith $";
#endif
/*
      Code for manipulating files.
*/
#include "files.h"

/*@C
   PetscGetWorkingDirectory - Get the current working directory.

   Input paramters:
.  path - use to hold the result value
.  len  - maximum length of path

.keywords, system, get, current, working, directory
@*/
int PetscGetWorkingDirectory( char *path,int len )
{
#if defined(PARCH_sun4)
  getwd( path );
#else
  getcwd( path, len );
#endif
  return 0;
}
