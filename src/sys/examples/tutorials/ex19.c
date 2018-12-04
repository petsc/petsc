
static char help[] = "Illustrates creating an options database.\n\n";

/*T
   Concepts: introduction to PETSc;
   Concepts: printing^in parallel
   Processors: n
T*/



#include <petscsys.h>
#include <petscviewer.h>
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscOptions   options;


  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsCreate(&options);CHKERRQ(ierr);
  ierr = PetscOptionsInsert(options,&argc,&argv,"optionsfile");CHKERRQ(ierr);
  ierr = PetscOptionsInsertString(options,"-option1 value1 -option2 -option3 value3");CHKERRQ(ierr);
  ierr = PetscOptionsView(options,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscOptionsDestroy(&options);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
     localrunfiles: optionsfile
     filter: grep -v malloc | grep -v nox | grep -v display | grep -v saws_port | grep -v vecscatter | grep -v options_left | grep -v check_pointer_intensity
TEST*/
