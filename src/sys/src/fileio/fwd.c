#ifndef lint
static char vcid[] = "$Id: fwd.c,v 1.1 1996/01/30 18:33:39 bsmith Exp bsmith $";
#endif
/*
      Code for manipulating files.
*/
#include "files.h"

/*@C
   SYGetwd - Get the current working directory.

   Input paramters:
.  path - use to hold the result value
.  len  - maximum length of path

.keywords, system, get, current, working, directory
@*/
int SYGetwd( char *path,int len )
{
#if defined(PARCH_sun4)
  getwd( path );
#else
  getcwd( path, len );
#endif
  return 0;
}
