#define PETSC_DESIRE_FEATURE_TEST_MACROS /* for lstat() */
#include <petscsys.h>
#include <sys/stat.h>
#if defined(PETSC_HAVE_DIRECT_H)
#include <direct.h>
#endif
#if defined(PETSC_HAVE_IO_H)
#include <io.h>
#endif
#if defined (PETSC_HAVE_STDINT_H)
#include <stdint.h>
#endif
#if defined(PETSC_HAVE_UNISTD_H) /* for mkdtemp */
#include <unistd.h>
#endif

PetscErrorCode PetscPathJoin(const char dname[],const char fname[],size_t n,char fullname[])
{
  size_t         l1,l2;
  PetscFunctionBegin;
  CHKERRQ(PetscStrlen(dname,&l1));
  CHKERRQ(PetscStrlen(fname,&l2));
  PetscCheckFalse((l1+l2+2)>n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Path length is greater than buffer size");
  CHKERRQ(PetscStrncpy(fullname,dname,n));
  CHKERRQ(PetscStrlcat(fullname,"/",n));
  CHKERRQ(PetscStrlcat(fullname,fname,n));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMkdir(const char dir[])
{
  int            err;
  PetscBool      flg;

  PetscFunctionBegin;
  CHKERRQ(PetscTestDirectory(dir,'w',&flg));
  if (flg) PetscFunctionReturn(0);
#if defined(PETSC_HAVE__MKDIR) && defined(PETSC_HAVE_DIRECT_H)
  err = _mkdir(dir);
#else
  err = mkdir(dir,S_IRWXU|S_IRGRP|S_IXGRP);
#endif
  PetscCheckFalse(err,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not create dir: %s",dir);
  PetscFunctionReturn(0);
}

#if defined(PETSC_USING_DARWIN)
/*
    Apple's mkdtemp() crashes under Valgrind so this replaces it with a version that does not crash under valgrind
*/
#include "apple_fdir.h"
#endif

/*@C
  PetscMkdtemp - Create a folder with a unique name given a filename template.

  Not Collective

  Input Parameters:
. dir - file name template, the last six characters must be 'XXXXXX', and they will be modified upon return

  Level: developer

.seealso: PetscMkdir()
@*/
PetscErrorCode PetscMkdtemp(char dir[])
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_WINDOWS_H) && defined(PETSC_HAVE_IO_H) && defined(PETSC_HAVE__MKDIR) && defined(PETSC_HAVE_DIRECT_H)
  {
    int            err = 1;
    char           name[PETSC_MAX_PATH_LEN];
    PetscInt       i = 0,max_retry = 26;
    size_t         len;
    PetscErrorCode ierr;

    while (err && i < max_retry) {
      CHKERRQ(PetscStrncpy(name,dir,sizeof(name)));
      CHKERRQ(PetscStrlen(name,&len));
      err = _mktemp_s(name,len+1);
      PetscCheckFalse(err,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not generate a unique name using the template: %s",dir);
      err = _mkdir(name);
      i++;
    }
    PetscCheckFalse(err,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Exceeds maximum retry time when creating temporary dir using the template: %s",dir);
    CHKERRQ(PetscStrncpy(dir,name,len+1));
  }
#else
  dir = mkdtemp(dir);
  PetscCheckFalse(!dir,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not create temporary dir");
#endif
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DIRECT_H)
PetscErrorCode PetscRMTree(const char dir[])
{
  PetscErrorCode ierr;
  struct _finddata_t data;
  char loc[PETSC_MAX_PATH_LEN];
  PetscBool flg1, flg2;
#if defined (PETSC_HAVE_STDINT_H)
  intptr_t handle;
#else
  long handle;
  #endif

  PetscFunctionBegin;
  CHKERRQ(PetscPathJoin(dir,"*",PETSC_MAX_PATH_LEN,loc));
  handle = _findfirst(loc, &data);
  if (handle == -1) {
    PetscBool flg;
    CHKERRQ(PetscTestDirectory(loc,'r',&flg));
    PetscCheckFalse(flg,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Cannot access directory to delete: %s",dir);
    CHKERRQ(PetscTestFile(loc,'r',&flg));
    PetscCheckFalse(flg,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Specified path is a file - not a dir: %s",dir);
    PetscFunctionReturn(0); /* perhaps the dir was not yet created */
  }
  while (_findnext(handle, &data) != -1) {
    CHKERRQ(PetscStrcmp(data.name, ".",&flg1));
    CHKERRQ(PetscStrcmp(data.name, "..",&flg2));
    if (flg1 || flg2) continue;
    CHKERRQ(PetscPathJoin(dir,data.name,PETSC_MAX_PATH_LEN,loc));
    if (data.attrib & _A_SUBDIR) {
      CHKERRQ(PetscRMTree(loc));
    } else{
      PetscCheckFalse(remove(loc),PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not delete file: %s",loc);
    }
  }
  _findclose(handle);
  PetscCheckFalse(_rmdir(dir),PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not delete dir: %s",dir);
  PetscFunctionReturn(0);
}
#else
#include <dirent.h>
#include <unistd.h>
PetscErrorCode PetscRMTree(const char dir[])
{
  struct dirent *data;
  char loc[PETSC_MAX_PATH_LEN];
  PetscBool flg1, flg2;
  DIR *dirp;
  struct stat statbuf;

  PetscFunctionBegin;
  dirp = opendir(dir);
  if (!dirp) {
    PetscBool flg;
    CHKERRQ(PetscTestDirectory(dir,'r',&flg));
    PetscCheckFalse(flg,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Cannot access directory to delete: %s",dir);
    CHKERRQ(PetscTestFile(dir,'r',&flg));
    PetscCheckFalse(flg,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Specified path is a file - not a dir: %s",dir);
    PetscFunctionReturn(0); /* perhaps the dir was not yet created */
  }
  while ((data = readdir(dirp))) {
    CHKERRQ(PetscStrcmp(data->d_name, ".",&flg1));
    CHKERRQ(PetscStrcmp(data->d_name, "..",&flg2));
    if (flg1 || flg2) continue;
    CHKERRQ(PetscPathJoin(dir,data->d_name,PETSC_MAX_PATH_LEN,loc));
    PetscCheckFalse(lstat(loc,&statbuf) <0,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"cannot run lstat() on: %s",loc);
    if (S_ISDIR(statbuf.st_mode)) {
      CHKERRQ(PetscRMTree(loc));
    } else {
      PetscCheckFalse(unlink(loc),PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not delete file: %s",loc);
    }
  }
  closedir(dirp);
  PetscCheckFalse(rmdir(dir),PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not delete dir: %s",dir);
  PetscFunctionReturn(0);
}
#endif
