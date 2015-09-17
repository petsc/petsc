#include <petscsys.h>
#include <direct.h>
#include <io.h>
#if defined (PETSC_HAVE_STDINT_H)
#include <stdint.h>
#endif

#undef __FUNCT__
#define __FUNCT__ "PetscPathJoin"
PetscErrorCode PetscPathJoin(const char dname[],const char fname[],size_t n,char fullname[])
{
  PetscErrorCode ierr;
  size_t l1,l2;
  PetscFunctionBegin;
  ierr = PetscStrlen(dname,&l1);CHKERRQ(ierr);
  ierr = PetscStrlen(fname,&l2);CHKERRQ(ierr);
  if ((l1+l2+2)>n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Path length is greater than buffer size");
  ierr = PetscStrcpy(fullname,dname);CHKERRQ(ierr);
  ierr = PetscStrcat(fullname,"/");CHKERRQ(ierr);
  ierr = PetscStrcat(fullname,fname);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscMkdir"
PetscErrorCode PetscMkdir(const char dir[])
{
  PetscFunctionBegin;
  if(_mkdir(dir)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not create dir: %s",dir);
     PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscRMTree"
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
  if(handle == -1) {
    PetscBool flg;
    ierr = PetscTestDirectory(loc,'r',&flg);CHKERRQ(ierr);
    if (flg) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Cannot access directory to delete: %s",dir);
    ierr = PetscTestFile(loc,'r',&flg);CHKERRQ(ierr);
    if (flg) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Specified path is a file - not a dir: %s",dir);
    PetscFunctionReturn(0); /* perhaps the dir was not yet created */
  }
  while(_findnext(handle, &data) != -1) {
    ierr = PetscStrcmp(data.name, ".",&flg1);CHKERRQ(ierr);
    ierr = PetscStrcmp(data.name, "..",&flg2);CHKERRQ(ierr);
    if (flg1 || flg2) continue;
    ierr = PetscPathJoin(dir,data.name,PETSC_MAX_PATH_LEN,loc);CHKERRQ(ierr);
    if(data.attrib & _A_SUBDIR) {
      ierr = PetscRMTree(loc);CHKERRQ(ierr);
    } else{
      if (remove(loc)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not delete file: %s",loc);
    }
  }
  _findclose(handle);
  if (_rmdir(dir)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Could not delete dir: %s",dir);
  PetscFunctionReturn(0);
}


