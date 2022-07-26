
static char help[] = "Illustrates creating an options database.\n\n";

#include <petscsys.h>
#include <petscviewer.h>
int main(int argc,char **argv)
{
  PetscOptions   options;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsCreate(&options));
  PetscCall(PetscOptionsInsert(options,&argc,&argv,"optionsfile"));
  PetscCall(PetscOptionsInsertString(options,"-option1 value1 -option2 -option3 value3"));
  PetscCall(PetscOptionsView(options,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscOptionsDestroy(&options));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     localrunfiles: optionsfile
     filter: egrep -v "(options_left)"

TEST*/
