static char help[] = "Example for PetscOptionsInsertFileYAML\n";
#include <petscsys.h>
#include <petscviewer.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  char            filename[PETSC_MAX_PATH_LEN];
  PetscBool       flg;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetString(NULL,NULL,"-f",filename,sizeof filename,&flg);
  if (flg) {
    ierr = PetscOptionsInsertFileYAML(PETSC_COMM_WORLD,filename,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscOptionsView(PETSC_NULL,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return(0);
}

