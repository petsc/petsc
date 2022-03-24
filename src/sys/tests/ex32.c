
static char help[] = "Tests deletion of mixed case options";

#include <petscsys.h>

int main(int argc,char **argv)
{

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  CHKERRQ(PetscOptionsSetValue(NULL,"-abc",NULL));
  CHKERRQ(PetscOptionsSetValue(NULL,"-FOO",NULL));
  CHKERRQ(PetscOptionsClearValue(NULL,"-FOO"));
  CHKERRQ(PetscOptionsView(NULL,NULL));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -skip_petscrc -options_left 0 -use_gpu_aware_mpi 0
      filter: egrep -v "(malloc|saws_port_auto_select|vecscatter_mpi1|error_output_stdout|check_pointer_intensity|cuda_initialize|nox|use_gpu_aware_mpi|checkstack)"
TEST*/
