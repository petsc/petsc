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
  PetscErrorCode ierr;
  size_t         l1,l2;
  PetscFunctionBegin;
  ierr = PetscStrlen(dname,&l1);CHKERRQ(ierr);
  ierr = PetscStrlen(fname,&l2);CHKERRQ(ierr);
  if ((l1+l2+2)>n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Path length is greater than buffer size");
  ierr = PetscStrncpy(fullname,dname,n);CHKERRQ(ierr);
  ierr = PetscStrlcat(fullname,"/",n);CHKERRQ(ierr);
  ierr = PetscStrlcat(fullname,fname,n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMkdir(const char dir[])
{
  int            err;
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscTestDirectory(dir,'w',&flg);CHKERRQ(ierr);
  if (flg) PetscFunctionReturn(0);
#if defined(PETSC_HAVE__MKDIR) && defined(PETSC_HAVE_DIRECT_H)
  err = _mkdir(dir);
#else
  err = mkdir(dir,S_IRWXU|S_IRGRP|S_IXGRP);
#endif
  if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not create dir: %s",dir);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_VALGRIND_DARWIN)
#include "apple_fdir.c"
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
      ierr = PetscStrncpy(name,dir,sizeof(name));CHKERRQ(ierr);
      ierr = PetscStrlen(name,&len);CHKERRQ(ierr);
      err = _mktemp_s(name,len+1);
      if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not generate a unique name using the template: %s",dir);
      err = _mkdir(name);
      i++;
    }
    if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Exceeds maximum retry time when creating temporary dir using the template: %s",dir);
    ierr = PetscStrncpy(dir,name,len+1);CHKERRQ(ierr);
  }
#else
  dir = mkdtemp(dir);
  if (!dir) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not create temporary dir using the template: %s",dir);
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
  ierr = PetscPathJoin(dir,"*",PETSC_MAX_PATH_LEN,loc);CHKERRQ(ierr);
  handle = _findfirst(loc, &data);
  if (handle == -1) {
    PetscBool flg;
    ierr = PetscTestDirectory(loc,'r',&flg);CHKERRQ(ierr);
    if (flg) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Cannot access directory to delete: %s",dir);
    ierr = PetscTestFile(loc,'r',&flg);CHKERRQ(ierr);
    if (flg) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Specified path is a file - not a dir: %s",dir);
    PetscFunctionReturn(0); /* perhaps the dir was not yet created */
  }
  while (_findnext(handle, &data) != -1) {
    ierr = PetscStrcmp(data.name, ".",&flg1);CHKERRQ(ierr);
    ierr = PetscStrcmp(data.name, "..",&flg2);CHKERRQ(ierr);
    if (flg1 || flg2) continue;
    ierr = PetscPathJoin(dir,data.name,PETSC_MAX_PATH_LEN,loc);CHKERRQ(ierr);
    if (data.attrib & _A_SUBDIR) {
      ierr = PetscRMTree(loc);CHKERRQ(ierr);
    } else{
      if (remove(loc)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not delete file: %s",loc);
    }
  }
  _findclose(handle);
  if (_rmdir(dir)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not delete dir: %s",dir);
  PetscFunctionReturn(0);
}
#else
#include <dirent.h>
#include <unistd.h>
PetscErrorCode PetscRMTree(const char dir[])
{
  PetscErrorCode ierr;
  struct dirent *data;
  char loc[PETSC_MAX_PATH_LEN];
  PetscBool flg1, flg2;
  DIR *dirp;
  struct stat statbuf;

  PetscFunctionBegin;
  dirp = opendir(dir);
  if (!dirp) {
    PetscBool flg;
    ierr = PetscTestDirectory(dir,'r',&flg);CHKERRQ(ierr);
    if (flg) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Cannot access directory to delete: %s",dir);
    ierr = PetscTestFile(dir,'r',&flg);CHKERRQ(ierr);
    if (flg) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Specified path is a file - not a dir: %s",dir);
    PetscFunctionReturn(0); /* perhaps the dir was not yet created */
  }
  while ((data = readdir(dirp))) {
    ierr = PetscStrcmp(data->d_name, ".",&flg1);CHKERRQ(ierr);
    ierr = PetscStrcmp(data->d_name, "..",&flg2);CHKERRQ(ierr);
    if (flg1 || flg2) continue;
    ierr = PetscPathJoin(dir,data->d_name,PETSC_MAX_PATH_LEN,loc);CHKERRQ(ierr);
    if (lstat(loc,&statbuf) <0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"cannot run lstat() on: %s",loc);
    if (S_ISDIR(statbuf.st_mode)) {
      ierr = PetscRMTree(loc);CHKERRQ(ierr);
    } else {
      if (unlink(loc)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not delete file: %s",loc);
    }
  }
  closedir(dirp);
  if (rmdir(dir)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not delete dir: %s",dir);
  PetscFunctionReturn(0);
}
#endif
