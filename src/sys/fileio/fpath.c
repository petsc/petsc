
#include <petscsys.h>
#if defined(PETSC_HAVE_PWD_H)
#include <pwd.h>
#endif

/*@C
   PetscGetFullPath - Given a filename, returns the fully qualified file name.

   Not Collective

   Input Parameters:
+  path     - pathname to qualify
.  fullpath - pointer to buffer to hold full pathname
-  flen     - size of fullpath

   Level: developer


.seealso: PetscGetRelativePath()
@*/
PetscErrorCode  PetscGetFullPath(const char path[],char fullpath[],size_t flen)
{
  PetscErrorCode ierr;
  size_t         ln;
  PetscBool      flg;

  PetscFunctionBegin;
  if (path[0] == '/') {
    ierr = PetscStrncmp("/tmp_mnt/",path,9,&flg);CHKERRQ(ierr);
    if (flg) {ierr = PetscStrncpy(fullpath,path + 8,flen);CHKERRQ(ierr);}
    else     {ierr = PetscStrncpy(fullpath,path,flen);CHKERRQ(ierr);}
    fullpath[flen-1] = 0;
    PetscFunctionReturn(0);
  }

  ierr = PetscStrncpy(fullpath,path,flen);CHKERRQ(ierr);
  fullpath[flen-1] = 0;
  /* Remove the various "special" forms (~username/ and ~/) */
  if (fullpath[0] == '~') {
    char tmppath[PETSC_MAX_PATH_LEN],*rest;
    if (fullpath[1] == '/') {
      ierr = PetscGetHomeDirectory(tmppath,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
      rest = fullpath + 2;
    } else {
#if defined(PETSC_HAVE_PWD_H)
      struct passwd  *pwde;
      char *p,*name;

      /* Find username */
      name = fullpath + 1;
      p    = name;
      while (*p && *p != '/') p++;
      *p   = 0;
      rest = p + 1;
      pwde = getpwnam(name);
      if (!pwde) PetscFunctionReturn(0);

      ierr = PetscStrcpy(tmppath,pwde->pw_dir);CHKERRQ(ierr);
#else
      PetscFunctionReturn(0);
#endif
    }
    ierr = PetscStrlen(tmppath,&ln);CHKERRQ(ierr);
    if (tmppath[ln-1] != '/') {ierr = PetscStrcat(tmppath+ln-1,"/");CHKERRQ(ierr);}
    ierr = PetscStrcat(tmppath,rest);CHKERRQ(ierr);
    ierr = PetscStrncpy(fullpath,tmppath,flen);CHKERRQ(ierr);
    fullpath[flen-1] = 0;
  } else {
    ierr = PetscGetWorkingDirectory(fullpath,flen);CHKERRQ(ierr);
    ierr = PetscStrlen(fullpath,&ln);CHKERRQ(ierr);
    ierr = PetscStrncpy(fullpath+ln,"/",flen - ln);CHKERRQ(ierr);
    fullpath[flen-1] = 0;
    ierr = PetscStrlen(fullpath,&ln);CHKERRQ(ierr);
    if (path[0] == '.' && path[1] == '/') {
      ierr = PetscStrlcat(fullpath,path+2,flen);CHKERRQ(ierr);
    } else {
      ierr = PetscStrlcat(fullpath,path,flen);CHKERRQ(ierr);
    }
    fullpath[flen-1] = 0;
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
