#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ffpath.c,v 1.12 1997/10/19 03:23:45 bsmith Exp bsmith $";
#endif
/*
      Code for manipulating files.
*/
#include "src/sys/src/files.h"

#undef __FUNC__  
#define __FUNC__ "PetscGetFileFromPath"
/*@C
   PetscGetFileFromPath - Finds a file from a name and a path string.  A 
   default can be provided.

   Input Parameters:
.  path - A string containing "directory:directory:..." (without the
	  quotes, of course).
	  As a special case, if the name is a single FILE, that file is
	  used.
.  defname - default name
.  name - file name to use with the directories from env
.  mode - file mode desired (usually 'r' for readable; 'w' for writable and
          'e' for executable are also supported)

   Output Parameter:
.  fname - qualified file name

   Returns:
   1 on success, 0 on failure.

.keywords: system, get, file, from, path

.seealso:
@*/
int PetscGetFileFromPath(char *path,char *defname,char *name,char *fname, char mode)
{
#if !defined(PARCH_nt)
  char       *p, *cdir, trial[MAXPATHLEN],*senv, *env;
  int        ln,ierr;
  PetscTruth flag;

  PetscFunctionBegin;
  /* Setup default */
  PetscGetFullPath(defname,fname,MAXPATHLEN);

  if (path) {
    /* Check to see if the path is a valid regular FILE */
    ierr = PetscTestFile( path, mode,&flag); CHKERRQ(ierr);
    if (flag) {
      PetscStrcpy( fname, path );
      PetscFunctionReturn(1);
    }
    
    /* Make a local copy of path and mangle it */
    senv = env = (char *)PetscMalloc( PetscStrlen(path) + 1 ); CHKPTRQ(senv);
    PetscStrcpy( env, path );
    while (env) {
      /* Find next directory in env */
      cdir = env;
      p    = PetscStrchr( env, ':' );
      if (p) {
	*p  = 0;
	env = p + 1;
      }
      else
	env = 0;

      /* Form trial file name */
      PetscStrcpy( trial, cdir );
      ln = PetscStrlen( trial );
      if (trial[ln-1] != '/')  trial[ln++] = '/';
	
      PetscStrcpy( trial + ln, name );

      ierr = PetscTestFile( path, mode,&flag); CHKERRQ(ierr);
      if (flag) {
        /* need PetscGetFullPath rather then copy in case path has . in it */
	PetscGetFullPath( trial,  fname, MAXPATHLEN );
	PetscFree( senv );
        PetscFunctionReturn(1);
      }
    }
    PetscFree( senv );
  }

  ierr = PetscTestFile( path, mode,&flag); CHKERRQ(ierr);
  if (flag) PetscFunctionReturn(1);
#endif
  PetscFunctionReturn(0);
}
