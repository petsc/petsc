static char help[] = "Example for PetscOptionsInsertFileYAML\n";


#include <petscsys.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  char            filename[PETSC_MAX_PATH_LEN];
  PetscBool       flg;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetString(NULL,NULL,"-f",filename,sizeof(filename),&flg);
  if (flg) {
    ierr = PetscOptionsInsertFileYAML(PETSC_COMM_WORLD,filename,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscOptionsView(NULL,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   build:
      requires: yaml

   test:
      suffix: 1
      requires: yaml
      args: -f petsc.yml
      filter:  grep -v saws_port_auto_select |grep -v malloc_dump | grep -v display
      localrunfiles: petsc.yml

   test:
      suffix: 2
      requires: yaml
      filter:  grep -v saws_port_auto_select
      args: -options_file_yaml petsc.yml |grep -v malloc_dump | grep -v display
      localrunfiles: petsc.yml

TEST*/
