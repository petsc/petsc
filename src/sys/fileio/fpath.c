
/*
      Code for opening and closing files.
*/
#include <petscsys.h>
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
#if defined(PETSC_HAVE_SYS_UTSNAME_H)
#include <sys/utsname.h>
#endif
#include <fcntl.h>
#if defined(PETSC_HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif

#if defined(PETSC_HAVE_PWD_H)

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
PetscErrorCode  PetscGetFullPath(const char path[],char fullpath[],size_t flen)
{
  struct passwd *pwde;
  PetscErrorCode ierr;
  size_t        ln;
  PetscBool     flg;

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
    char tmppath[PETSC_MAX_PATH_LEN];
    if (fullpath[1] == '/') {
#if defined(PETSC_HAVE_GETPWUID)
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
	while (*p && *p != '/') p++;
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
    char tmppath[PETSC_MAX_PATH_LEN];
    ierr = PetscStrcpy(tmppath,fullpath + 8);CHKERRQ(ierr);
    ierr = PetscStrcpy(fullpath,tmppath);CHKERRQ(ierr);
  }
  /* We could try to handle things like the removal of .. etc */
  PetscFunctionReturn(0);
}
#elif defined(PETSC_HAVE__FULLPATH)
#undef __FUNCT__  
#define __FUNCT__ "PetscGetFullPath"
PetscErrorCode  PetscGetFullPath(const char path[],char fullpath[],size_t flen)
{
  PetscFunctionBegin;
  _fullpath(fullpath,path,flen);
  PetscFunctionReturn(0);
}
#else
#undef __FUNCT__  
#define __FUNCT__ "PetscGetFullPath"
PetscErrorCode  PetscGetFullPath(const char path[],char fullpath[],size_t flen)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrcpy(fullpath,path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}	
#endif
