
static char help[] = "Tests deletion of mixed case options";

#include <petscsys.h>

int main(int argc,char **argv)
{

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCall(PetscOptionsSetValue(NULL,"-abc",NULL));
  PetscCall(PetscOptionsSetValue(NULL,"-FOO",NULL));
  PetscCall(PetscOptionsClearValue(NULL,"-FOO"));
  PetscCall(PetscOptionsView(NULL,NULL));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -skip_petscrc -options_left 0 -use_gpu_aware_mpi 0

TEST*/
