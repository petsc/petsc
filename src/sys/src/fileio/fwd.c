#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fwd.c,v 1.12 1997/09/05 18:22:21 gropp Exp bsmith $";
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
