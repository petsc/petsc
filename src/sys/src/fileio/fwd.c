#ifndef lint
static char vcid[] = "$Id: file.c,v 1.27 1995/12/31 17:17:34 curfman Exp bsmith $";
#endif
/*
      Code for manipulating files.
*/
#include "file.h"

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
