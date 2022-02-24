
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
  size_t         ln;
  PetscBool      flg;

  PetscFunctionBegin;
  if (path[0] == '/') {
    CHKERRQ(PetscStrncmp("/tmp_mnt/",path,9,&flg));
    if (flg) CHKERRQ(PetscStrncpy(fullpath,path + 8,flen));
    else     CHKERRQ(PetscStrncpy(fullpath,path,flen));
    fullpath[flen-1] = 0;
    PetscFunctionReturn(0);
  }

  CHKERRQ(PetscStrncpy(fullpath,path,flen));
  fullpath[flen-1] = 0;
  /* Remove the various "special" forms (~username/ and ~/) */
  if (fullpath[0] == '~') {
    char tmppath[PETSC_MAX_PATH_LEN],*rest;
    if (fullpath[1] == '/') {
      CHKERRQ(PetscGetHomeDirectory(tmppath,PETSC_MAX_PATH_LEN));
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

      CHKERRQ(PetscStrcpy(tmppath,pwde->pw_dir));
#else
      PetscFunctionReturn(0);
#endif
    }
    CHKERRQ(PetscStrlen(tmppath,&ln));
    if (tmppath[ln-1] != '/') CHKERRQ(PetscStrcat(tmppath+ln-1,"/"));
    CHKERRQ(PetscStrcat(tmppath,rest));
    CHKERRQ(PetscStrncpy(fullpath,tmppath,flen));
    fullpath[flen-1] = 0;
  } else {
    CHKERRQ(PetscGetWorkingDirectory(fullpath,flen));
    CHKERRQ(PetscStrlen(fullpath,&ln));
    CHKERRQ(PetscStrncpy(fullpath+ln,"/",flen - ln));
    fullpath[flen-1] = 0;
    CHKERRQ(PetscStrlen(fullpath,&ln));
    if (path[0] == '.' && path[1] == '/') {
      CHKERRQ(PetscStrlcat(fullpath,path+2,flen));
    } else {
      CHKERRQ(PetscStrlcat(fullpath,path,flen));
    }
    fullpath[flen-1] = 0;
  }

  /* Remove the automounter part of the path */
  CHKERRQ(PetscStrncmp(fullpath,"/tmp_mnt/",9,&flg));
  if (flg) {
    char tmppath[PETSC_MAX_PATH_LEN];
    CHKERRQ(PetscStrcpy(tmppath,fullpath + 8));
    CHKERRQ(PetscStrcpy(fullpath,tmppath));
  }
  /* We could try to handle things like the removal of .. etc */
  PetscFunctionReturn(0);
}
