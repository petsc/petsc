static char help[] = "Demonstrates PETSc path routines.\n";

#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "CheckLen"
static PetscErrorCode CheckLen(const char *path,size_t len,size_t *used)
{
  char           *buf,cmd[4096];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* dynamically allocate so valgrind and PETSc can check for overflow */
  ierr = PetscMalloc(len,&buf);CHKERRQ(ierr);
  ierr = PetscGetFullPath(path,buf,len);CHKERRQ(ierr);
  ierr = PetscSNPrintf(cmd,sizeof cmd,"test $(realpath %s | head -c %d) = %s",path,(int)len-1,buf);CHKERRQ(ierr);
  if (system(cmd)) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"$(realpath %s | head -c %d) != %s\n",path,(int)len-1,buf);CHKERRQ(ierr);
  }
  if (used) {ierr = PetscStrlen(buf,used);CHKERRQ(ierr);}
  ierr = PetscFree(buf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Check"
static PetscErrorCode Check(const char *path)
{
  PetscErrorCode ierr;
  size_t         used;

  PetscFunctionBegin;
  ierr = CheckLen(path,PETSC_MAX_PATH_LEN,&used);CHKERRQ(ierr);
  ierr = CheckLen(path,used-1,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  char           user[256],buf[512];

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = Check("~/file-name");CHKERRQ(ierr);
  ierr = PetscGetUserName(user,256);CHKERRQ(ierr);
  ierr = PetscSNPrintf(buf,sizeof buf,"~%s/file-name",user);CHKERRQ(ierr);
  ierr = Check(buf);CHKERRQ(ierr);
  ierr = Check("/dev/null");CHKERRQ(ierr);
  ierr = Check("./this-dir");CHKERRQ(ierr);
  ierr = Check("also-this-dir");CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

