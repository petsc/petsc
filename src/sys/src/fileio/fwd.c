#ifndef lint
static char vcid[] = "$Id: fwd.c,v 1.5 1996/08/08 14:41:26 bsmith Exp balay $";
#endif
/*
      Code for manipulating files.
*/
#include "src/sys/src/files.h"

#undef __FUNCTION__  
#define __FUNCTION__ "PetscGetWorkingDirectory"
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
