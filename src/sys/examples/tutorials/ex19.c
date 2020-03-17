
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
  CHKERRQ(PetscOptionsCreate(&options));
  CHKERRQ(PetscOptionsInsert(options,&argc,&argv,"optionsfile"));
  CHKERRQ(PetscOptionsInsertString(options,"-option1 value1 -option2 -option3 value3"));
  CHKERRQ(PetscOptionsView(options,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscOptionsDestroy(&options));
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
     localrunfiles: optionsfile
     filter: egrep -v "(malloc|nox|display|saws_port|vecscatter|options_left|check_pointer_intensity|cuda_initialize|error_output_stdout)"

TEST*/
