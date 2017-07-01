      program ex10f90

#include "petsc/finclude/petsc.h"
      use petsc
      implicit none

      PetscErrorCode                            :: ierr
      Character(len=256)                        :: filename
      PetscBool                                 :: flg

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      call PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-f',filename,flg,ierr);CHKERRA(ierr)
      if (flg) then
         call PetscOptionsInsertFileYAML(PETSC_COMM_WORLD,filename,PETSC_TRUE,ierr);CHKERRA(ierr)
      end if
      call PetscOptionsView(PETSC_NULL_OPTIONS,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)
      Call PetscFinalize(ierr)
      end program ex10f90



!
!/*TEST
!
!   build:
!     requires: yaml
!
!   test:
!      suffix: 1
!      args: -f petsc.yml
!      filter:   grep -v saws_port_auto_select |grep -v malloc_dump | grep -v display
!      localrunfiles: petsc.yml
!
!   test:
!      suffix: 2
!      args: -options_file_yaml petsc.yml
!      filter:   grep -v saws_port_auto_select |grep -v malloc_dump | grep -v display
!      localrunfiles: petsc.yml
!
!TEST*/
