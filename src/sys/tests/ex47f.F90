! Example for PetscOptionsInsertFileYAML: Fortran Example

program main

#include <petsc/finclude/petscsys.h>
      use petscsys

      implicit none
      PetscErrorCode                    :: ierr
      character(len=PETSC_MAX_PATH_LEN) :: filename
      PetscBool                         ::  flg

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr /= 0) then
        write(6,*)'Unable to initialize PETSc'
        stop
      endif

      call PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-f",filename,flg,ierr)
      if (flg) then
        call PetscOptionsInsertFileYAML(PETSC_COMM_WORLD,PETSC_NULL_OPTIONS,filename,PETSC_TRUE,ierr)
      end if

      call  PetscOptionsView(PETSC_NULL_OPTIONS,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call  PetscFinalize(ierr)

!/*TEST
!
! testset:
!   filter: egrep -v "(options_left|malloc_dump|malloc_test|saws_port_auto_select|display|check_pointer_intensity|error_output_stdout|nox|vecscatter_mpi1|checkstack|use_gpu_aware_mpi)"
!
!   test:
!      suffix: 1
!      args: -f petsc.yml -options_left 0
!      localrunfiles: petsc.yml
!      output_file: output/ex47_1.out
!
!   test:
!      suffix: 2
!      args: -options_file_yaml petsc.yml -options_left 0
!      localrunfiles: petsc.yml
!      output_file: output/ex47_2.out
!
!TEST*/
end program main
