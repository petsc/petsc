      program ex10f90

#include "petsc/finclude/petsc.h"
      use petsc
      implicit none

      PetscErrorCode                            :: ierr
      Character(len=256)                        :: filename
      PetscBool                                 :: flg
      PetscInt                                  :: n

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      call PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-f',filename,flg,ierr);CHKERRA(ierr)
      if (flg) then
         call PetscOptionsInsertFileYAML(PETSC_COMM_WORLD,PETSC_NULL_OPTIONS,filename,PETSC_TRUE,ierr);CHKERRA(ierr)
      end if
      call PetscOptionsView(PETSC_NULL_OPTIONS,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)
      call PetscOptionsAllUsed(PETSC_NULL_OPTIONS,n,ierr);CHKERRA(ierr);
      Call PetscFinalize(ierr)
      end program ex10f90

!
!/*TEST
!
! testset:
!   filter: egrep -v "(options_left|malloc_dump|malloc_test|saws_port_auto_select|display|check_pointer_intensity|error_output_stdout|nox|vecscatter_mpi1|use_gpu_aware_mpi|checkstack)"
!
!   test:
!      suffix: 1
!      args: -f petsc.yml -options_left 0
!      localrunfiles: petsc.yml
!
!   test:
!      suffix: 2
!      args: -options_file_yaml petsc.yml -options_left 0
!      localrunfiles: petsc.yml
!
!TEST*/
