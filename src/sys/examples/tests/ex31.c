
static char help[] = "Tests PetscGetFullPath().\n\n";

#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  char           fpath[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscGetFullPath("~/somefile",fpath,sizeof(fpath));CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s\n",fpath);CHKERRQ(ierr);
  ierr = PetscGetFullPath("someotherfile",fpath,sizeof(fpath));CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s\n",fpath);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

