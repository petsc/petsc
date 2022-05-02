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
  PetscCall(PetscStrlen(dname,&l1));
  PetscCall(PetscStrlen(fname,&l2));
  PetscCheck((l1+l2+2)<=n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Path length is greater than buffer size");
  PetscCall(PetscStrncpy(fullname,dname,n));
  PetscCall(PetscStrlcat(fullname,"/",n));
  PetscCall(PetscStrlcat(fullname,fname,n));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMkdir(const char dir[])
{
  int            err;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCall(PetscTestDirectory(dir,'w',&flg));
  if (flg) PetscFunctionReturn(0);
#if defined(PETSC_HAVE__MKDIR) && defined(PETSC_HAVE_DIRECT_H)
  err = _mkdir(dir);
#else
  err = mkdir(dir,S_IRWXU|S_IRGRP|S_IXGRP);
#endif
  PetscCheck(!err,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not create dir: %s",dir);
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

.seealso: `PetscMkdir()`
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

    while (err && i < max_retry) {
      PetscCall(PetscStrncpy(name,dir,sizeof(name)));
      PetscCall(PetscStrlen(name,&len));
      err = _mktemp_s(name,len+1);
      PetscCheck(!err,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not generate a unique name using the template: %s",dir);
      err = _mkdir(name);
      i++;
    }
    PetscCheck(!err,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Exceeds maximum retry time when creating temporary dir using the template: %s",dir);
    PetscCall(PetscStrncpy(dir,name,len+1));
  }
#else
  dir = mkdtemp(dir);
  PetscCheck(dir,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not create temporary dir");
#endif
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DIRECT_H)
PetscErrorCode PetscRMTree(const char dir[])
{
  struct _finddata_t data;
  char loc[PETSC_MAX_PATH_LEN];
  PetscBool flg1, flg2;
#if defined (PETSC_HAVE_STDINT_H)
  intptr_t handle;
#else
  long handle;
  #endif

  PetscFunctionBegin;
  PetscCall(PetscPathJoin(dir,"*",PETSC_MAX_PATH_LEN,loc));
  handle = _findfirst(loc, &data);
  if (handle == -1) {
    PetscBool flg;
    PetscCall(PetscTestDirectory(loc,'r',&flg));
    PetscCheck(!flg,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Cannot access directory to delete: %s",dir);
    PetscCall(PetscTestFile(loc,'r',&flg));
    PetscCheck(!flg,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Specified path is a file - not a dir: %s",dir);
    PetscFunctionReturn(0); /* perhaps the dir was not yet created */
  }
  while (_findnext(handle, &data) != -1) {
    PetscCall(PetscStrcmp(data.name, ".",&flg1));
    PetscCall(PetscStrcmp(data.name, "..",&flg2));
    if (flg1 || flg2) continue;
    PetscCall(PetscPathJoin(dir,data.name,PETSC_MAX_PATH_LEN,loc));
    if (data.attrib & _A_SUBDIR) {
      PetscCall(PetscRMTree(loc));
    } else{
      PetscCheck(!remove(loc),PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not delete file: %s",loc);
    }
  }
  _findclose(handle);
  PetscCheck(!_rmdir(dir),PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not delete dir: %s",dir);
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
    PetscCall(PetscTestDirectory(dir,'r',&flg));
    PetscCheck(!flg,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Cannot access directory to delete: %s",dir);
    PetscCall(PetscTestFile(dir,'r',&flg));
    PetscCheck(!flg,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Specified path is a file - not a dir: %s",dir);
    PetscFunctionReturn(0); /* perhaps the dir was not yet created */
  }
  while ((data = readdir(dirp))) {
    PetscCall(PetscStrcmp(data->d_name, ".",&flg1));
    PetscCall(PetscStrcmp(data->d_name, "..",&flg2));
    if (flg1 || flg2) continue;
    PetscCall(PetscPathJoin(dir,data->d_name,PETSC_MAX_PATH_LEN,loc));
    PetscCheck(lstat(loc,&statbuf) >=0,PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"cannot run lstat() on: %s",loc);
    if (S_ISDIR(statbuf.st_mode)) {
      PetscCall(PetscRMTree(loc));
    } else {
      PetscCheck(!unlink(loc),PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not delete file: %s",loc);
    }
  }
  closedir(dirp);
  PetscCheck(!rmdir(dir),PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not delete dir: %s",dir);
  PetscFunctionReturn(0);
}
#endif
