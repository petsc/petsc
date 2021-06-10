
static char help[] = "Tests options file parsing.\n\n";

#include <petscsys.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc,&argv,"ex54options",help);if (ierr) return ierr;
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 0
      localrunfiles: ex54options
      args: -options_left 0 -options_view
      filter: egrep -v "(malloc|nox|display|saws_port|vecscatter|check_pointer_intensity|cuda_initialize|error_output_stdout|use_gpu_aware_mpi|checkstack)" | egrep "^-"
   testset:
      args: -options_left 0 -options_view
      test:
        suffix: 1
        localrunfiles: ex54options ex54options_1a_wrong ex54options_1b_wrong ex54options_1c_wrong ex54options_1d_wrong ex54options_1e_wrong ex54options_1f_wrong ex54options_1g_wrong
        args: -options_file {{ex54options_1a_wrong ex54options_1b_wrong ex54options_1c_wrong ex54options_1d_wrong ex54options_1e_wrong ex54options_1f_wrong ex54options_1g_wrong}separate output}
        filter: Error: egrep -v "(malloc|nox|display|saws_port|vecscatter|check_pointer_intensity|cuda_initialize|error_output_stdout|use_gpu_aware_mpi|checkstack)" | egrep "ex54options"
      test:
        suffix: 1_options_file-ex54options_1h
        localrunfiles: ex54options ex54options_1h
        args: -options_file ex54options_1h
        filter: egrep -v "(malloc|nox|display|saws_port|vecscatter|check_pointer_intensity|cuda_initialize|error_output_stdout|use_gpu_aware_mpi|checkstack)" | egrep "^-"

TEST*/
