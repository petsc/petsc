/*$Id: ffpath.c,v 1.36 2001/03/23 23:20:30 balay Exp $*/

#include "petscconfig.h"
#include "petsc.h"
#include "petscsys.h"
#if defined(PETSC_HAVE_PWD_H)
#include <pwd.h>
#endif
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if !defined(PARCH_win32)
#include <sys/utsname.h>
#endif
#if defined(PARCH_win32)
#include <windows.h>
#include <io.h>
#include <direct.h>
#endif
#if defined (PARCH_win32_gnu)
#include <windows.h>
#endif
#if defined(PETSC_HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif
#include "petscfix.h"

#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscGetFileFromPath"
/*@C
   PetscGetFileFromPath - Finds a file from a name and a path string.  A 
                          default can be provided.

   Not Collective

   Input Parameters:
+  path - A string containing "directory:directory:..." (without the
	  quotes, of course).
	  As a special case, if the name is a single FILE, that file is
	  used.
.  defname - default name
.  name - file name to use with the directories from env
-  mode - file mode desired (usually r for readable, w for writable, or e for
          executable)

   Output Parameter:
.  fname - qualified file name

   Level: developer

   Concepts: files^finding in path
   Concepts: path^searching for file

@*/
int PetscGetFileFromPath(char *path,char *defname,char *name,char *fname,char mode)
{
#if !defined(PARCH_win32)
  char       *p,*cdir,trial[MAXPATHLEN],*senv,*env;
  int        ln,ierr;
  PetscTruth flg;

  PetscFunctionBegin;
  /* Setup default */
  ierr = PetscGetFullPath(defname,fname,MAXPATHLEN);CHKERRQ(ierr);

  if (path) {
    /* Check to see if the path is a valid regular FILE */
    ierr = PetscTestFile(path,mode,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscStrcpy(fname,path);CHKERRQ(ierr);
      PetscFunctionReturn(1);
    }
    
    /* Make a local copy of path and mangle it */
    ierr = PetscStrallocpy(path,&senv);CHKERRQ(ierr);
    env  = senv;
    while (env) {
      /* Find next directory in env */
      cdir = env;
      ierr = PetscStrchr(env,':',&p);CHKERRQ(ierr);
      if (p) {
	*p  = 0;
	env = p + 1;
      } else
	env = 0;

      /* Form trial file name */
      ierr = PetscStrcpy(trial,cdir);CHKERRQ(ierr);
      ierr = PetscStrlen(trial,&ln);CHKERRQ(ierr);
      if (trial[ln-1] != '/')  trial[ln++] = '/';
	
      ierr = PetscStrcpy(trial + ln,name);CHKERRQ(ierr);

      ierr = PetscTestFile(path,mode,&flg);CHKERRQ(ierr);
      if (flg) {
        /* need PetscGetFullPath rather then copy in case path has . in it */
	ierr = PetscGetFullPath(trial,fname,MAXPATHLEN);CHKERRQ(ierr);
	ierr = PetscFree(senv);CHKERRQ(ierr);
        PetscFunctionReturn(1);
      }
    }
    ierr = PetscFree(senv);CHKERRQ(ierr);
  }

  ierr = PetscTestFile(path,mode,&flg);CHKERRQ(ierr);
  if (flg) PetscFunctionReturn(1);
#endif
  PetscFunctionReturn(0);
}
