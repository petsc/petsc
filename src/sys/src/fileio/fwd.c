#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fwd.c,v 1.9 1997/02/27 00:44:07 balay Exp balay $";
#endif
/*
      Code for manipulating files.
*/
#include "src/sys/src/files.h"

#undef __FUNC__  
#define __FUNC__ "PetscGetWorkingDirectory" /* ADIC Ignore */
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
#elif defined(PARCH_nt)
  _getcwd( path, len );
#else
  getcwd( path, len );
#endif
  return 0;
}
