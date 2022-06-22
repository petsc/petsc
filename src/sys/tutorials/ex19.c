
static char help[] = "Illustrates creating an options database.\n\n";

#include <petscsys.h>
#include <petscviewer.h>
int main(int argc,char **argv)
{
  PetscOptions   options;

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
     filter: egrep -v "(malloc|nox|display|saws_port|vecscatter|options_left|check_pointer_intensity|cuda_initialize|error_output_stdout|use_gpu_aware_mpi|checkstack|checkfunctionlist)"

TEST*/
