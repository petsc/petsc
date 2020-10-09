
static char help[] = "Tests calling PetscOptionsSetValue() before PetscInitialize()\n\n";

/*T
   Concepts: introduction to PETSc;
   Concepts: printing^in parallel
   Processors: n
T*/

#include <petscsys.h>
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;

  /*
    Every PETSc routine should begin with the PetscInitialize() routine.
    argc, argv - These command line arguments are taken to extract the options
                 supplied to PETSc and options supplied to MPI.
    help       - When PETSc executable is invoked with the option -help,
                 it prints the various options that can be applied at
                 runtime.  The user can use the "help" variable place
                 additional help messages in this printout.

    Since when PetscInitialize() returns with an error the PETSc data structures
    may not be set up hence we cannot call CHKERRQ() hence directly return the error code.

    Since PetscOptionsSetValue() is called before the PetscInitialize() we cannot call
    CHKERRQ() on the error code and just return it directly.
  */
  ierr = PetscOptionsSetValue(NULL,"-no_signal_handler","true");if (ierr) return ierr;
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of processors = %d, rank = %d\n",size,rank);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      requires: define(PETSC_USE_LOG)
      nsize: 2
      args: -options_view -get_total_flops
      filter: egrep -v "(cuda_initialize|malloc|display|nox|Total flops|saws_port_auto_select|vecscatter_mpi1|options_left|error_output_stdout|check_pointer_intensity|use_gpu_aware_mpi)"

TEST*/
