#ifndef lint
static char vcid[] = "$Id: fwd.c,v 1.3 1996/03/19 21:24:22 bsmith Exp curfman $";
#endif
/*
      Code for manipulating files.
*/
#include "files.h"

/*@C
   PetscGetWorkingDirectory - Gets the current working directory.

   Input Paramters:
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
