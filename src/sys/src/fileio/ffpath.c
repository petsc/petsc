#ifndef lint
static char vcid[] = "$Id: ffpath.c,v 1.1 1996/01/30 18:32:25 bsmith Exp bsmith $";
#endif
/*
      Code for manipulating files.
*/
#include "files.h"

extern int SYiTestFile(char *,char,uid_t,gid_t);

/*@C
   SYGetFileFromPath - Finds a file from a name and a path string.  A 
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
int SYGetFileFromPath(char *path,char *defname,char *name,char *fname, char mode)
{
  char   *p, *cdir, trial[MAXPATHLEN],*senv, *env;
  int    ln;
  uid_t  uid;
  gid_t  gid;

  /* Setup default */
  SYGetFullPath(defname,fname,MAXPATHLEN);

  /* Get the (effective) user and group of the caller */
  uid = geteuid();
  gid = getegid();

  if (path) {
    /* Check to see if the path is a valid regular FILE */
    if (SYiTestFile( path, mode, uid, gid )) {
      PetscStrcpy( fname, path );
      return 1;
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

      if (SYiTestFile( trial, mode, uid, gid )) {
        /* need SYGetFullPath rather then copy in case path has . in it */
	SYGetFullPath( trial,  fname, MAXPATHLEN );
	PetscFree( senv );
	return 1;
      }
    }
    PetscFree( senv );
  }

  if (SYiTestFile( fname, mode, uid, gid )) return 1;
  else return 0;
}
