/*$Id: fpath.c,v 1.39 2001/03/23 23:20:30 balay Exp $*/
/*
      Code for opening and closing files.
*/
#include "petsc.h"
#include "petscsys.h"
#if defined(HAVE_PWD_H)
#include <pwd.h>
#endif
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#if defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(HAVE_STDLIB_H)
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
#include <fcntl.h>
#if defined(HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif
#include "petscfix.h"

#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif

#if defined(HAVE_PWD_H)

#undef __FUNCT__  
#define __FUNCT__ "PetscGetFullPath"
/*@C
   PetscGetFullPath - Given a filename, returns the fully qualified file name.

   Not Collective

   Input Parameters:
+  path     - pathname to qualify
.  fullpath - pointer to buffer to hold full pathname
-  flen     - size of fullpath

   Level: developer

   Concepts: full path
   Concepts: path^full

.seealso: PetscGetRelativePath()
@*/
int PetscGetFullPath(const char path[],char fullpath[],int flen)
{
  struct passwd *pwde;
  int           ierr,ln;
  PetscTruth    flg;

  PetscFunctionBegin;
  if (path[0] == '/') {
    ierr = PetscStrncmp("/tmp_mnt/",path,9,&flg);CHKERRQ(ierr);
    if (flg) {ierr = PetscStrncpy(fullpath,path + 8,flen);CHKERRQ(ierr);}
    else      {ierr = PetscStrncpy(fullpath,path,flen);CHKERRQ(ierr);}
    PetscFunctionReturn(0);
  }
  ierr = PetscGetWorkingDirectory(fullpath,flen);CHKERRQ(ierr);
  ierr = PetscStrlen(fullpath,&ln);CHKERRQ(ierr);
  ierr = PetscStrncat(fullpath,"/",flen - ln);CHKERRQ(ierr);
  if (path[0] == '.' && path[1] == '/') {
    ierr = PetscStrlen(fullpath,&ln);CHKERRQ(ierr);
    ierr = PetscStrncat(fullpath,path+2,flen - ln - 1);CHKERRQ(ierr);
  } else {
    ierr = PetscStrlen(fullpath,&ln);CHKERRQ(ierr);
    ierr = PetscStrncat(fullpath,path,flen - ln - 1);CHKERRQ(ierr);
  }

  /* Remove the various "special" forms (~username/ and ~/) */
  if (fullpath[0] == '~') {
    char tmppath[MAXPATHLEN];
    if (fullpath[1] == '/') {
#if !defined(MISSING_GETPWUID)
	pwde = getpwuid(geteuid());
	if (!pwde) PetscFunctionReturn(0);
	ierr = PetscStrcpy(tmppath,pwde->pw_dir);CHKERRQ(ierr);
	ierr = PetscStrlen(tmppath,&ln);CHKERRQ(ierr);
	if (tmppath[ln-1] != '/') {ierr = PetscStrcat(tmppath+ln-1,"/");CHKERRQ(ierr);}
	ierr = PetscStrcat(tmppath,fullpath + 2);CHKERRQ(ierr);
	ierr = PetscStrncpy(fullpath,tmppath,flen);CHKERRQ(ierr);
#else
        PetscFunctionReturn(0);
#endif
    } else {
	char *p,*name;

	/* Find username */
	name = fullpath + 1;
	p    = name;
	while (*p && isalnum((int)(*p))) p++;
	*p = 0; p++;
	pwde = getpwnam(name);
	if (!pwde) PetscFunctionReturn(0);
	
	ierr = PetscStrcpy(tmppath,pwde->pw_dir);CHKERRQ(ierr);
	ierr = PetscStrlen(tmppath,&ln);CHKERRQ(ierr);
	if (tmppath[ln-1] != '/') {ierr = PetscStrcat(tmppath+ln-1,"/");CHKERRQ(ierr);}
	ierr = PetscStrcat(tmppath,p);CHKERRQ(ierr);
	ierr = PetscStrncpy(fullpath,tmppath,flen);CHKERRQ(ierr);
    }
  }
  /* Remove the automounter part of the path */
  ierr = PetscStrncmp(fullpath,"/tmp_mnt/",9,&flg);CHKERRQ(ierr);
  if (flg) {
    char tmppath[MAXPATHLEN];
    ierr = PetscStrcpy(tmppath,fullpath + 8);CHKERRQ(ierr);
    ierr = PetscStrcpy(fullpath,tmppath);CHKERRQ(ierr);
  }
  /* We could try to handle things like the removal of .. etc */
  PetscFunctionReturn(0);
}
#elif defined (PARCH_win32)
#undef __FUNCT__  
#define __FUNCT__ "PetscGetFullPath"
int PetscGetFullPath(const char path[],char fullpath[],int flen)
{
  PetscFunctionBegin;
  _fullpath(fullpath,path,flen);
  PetscFunctionReturn(0);
}
#else
#undef __FUNCT__  
#define __FUNCT__ "PetscGetFullPath"
int PetscGetFullPath(const char path[],char fullpath[],int flen)
{
  int ierr;

  PetscFunctionBegin;
  ierr = PetscStrcpy(fullpath,path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}	
#endif
