#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fwd.c,v 1.13 1997/10/19 03:23:45 bsmith Exp bsmith $";
#endif
/*
      Code for manipulating files.
*/
#include "src/sys/src/files.h"

#undef __FUNC__  
#define __FUNC__ "PetscGetWorkingDirectory"
/*@C
   PetscGetWorkingDirectory - Gets the current working directory.

   Input Paramters:
.  path - use to hold the result value
.  len  - maximum length of path

   Not Collective

.keywords, system, get, current, working, directory
@*/
int PetscGetWorkingDirectory( char *path,int len )
{
  PetscFunctionBegin;
#if defined(HAVE_GETWD)
  getwd( path );
#elif defined(PARCH_nt)
  _getcwd( path, len );
#else
  getcwd( path, len );
#endif
  PetscFunctionReturn(0);
}
