#ifndef lint
static char vcid[] = "$Id: file.c,v 1.27 1995/12/31 17:17:34 curfman Exp bsmith $";
#endif
/*
      Code for opening and closing files.
*/
#include "file.h"

#if defined(HAVE_PWD_H)
/*@C
   SYGetFullPath - Given a filename, returns the fully qualified file name.

   Input Parameters:
.  path     - pathname to qualify
.  fullpath - pointer to buffer to hold full pathname
.  flen     - size of fullpath

.keywords: system, get, full, path

.seealso: SYGetRelativePath()
@*/
int SYGetFullPath( char *path, char *fullpath, int flen )
{
  struct passwd *pwde;
  int           ln;

  if (path[0] == '/') {
    if (PetscStrncmp("/tmp_mnt/",path,9) == 0) PetscStrncpy(fullpath, path + 8, flen);
    else PetscStrncpy( fullpath, path, flen); 
    return 0;
  }
  SYGetwd( fullpath, flen );
  PetscStrncat( fullpath,"/",flen - PetscStrlen(fullpath) );
  if ( path[0] == '.' && path[1] == '/' ) 
    PetscStrncat( fullpath, path+2, flen - PetscStrlen(fullpath) - 1 );
  else 
    PetscStrncat( fullpath, path, flen - PetscStrlen(fullpath) - 1 );

  /* Remove the various "special" forms (~username/ and ~/) */
  if (fullpath[0] == '~') {
    char tmppath[MAX_FILE_NAME];
    if (fullpath[1] == '/') {
	pwde = getpwuid( geteuid() );
	if (!pwde) return 0;
	PetscStrcpy( tmppath, pwde->pw_dir );
	ln = PetscStrlen( tmppath );
	if (tmppath[ln-1] != '/') PetscStrcat( tmppath+ln-1, "/" );
	PetscStrcat( tmppath, fullpath + 2 );
	PetscStrncpy( fullpath, tmppath, flen );
    }
    else {
	char *p, *name;

	/* Find username */
	name = fullpath + 1;
	p    = name;
	while (*p && isalnum(*p)) p++;
	*p = 0; p++;
	pwde = getpwnam( name );
	if (!pwde) return 0;
	
	PetscStrcpy( tmppath, pwde->pw_dir );
	ln = PetscStrlen( tmppath );
	if (tmppath[ln-1] != '/') PetscStrcat( tmppath+ln-1, "/" );
	PetscStrcat( tmppath, p );
	PetscStrncpy( fullpath, tmppath, flen );
    }
  }
  /* Remove the automounter part of the path */
  if (PetscStrncmp( fullpath, "/tmp_mnt/", 9 ) == 0) {
    char tmppath[MAX_FILE_NAME];
    PetscStrcpy( tmppath, fullpath + 8 );
    PetscStrcpy( fullpath, tmppath );
  }
  /* We could try to handle things like the removal of .. etc */
  return 0;
}
#else
int SYGetFullPath( char *path, char *fullpath, int flen )
{
  PetscStrcpy( fullpath, path );
  return 0;
}	
#endif
