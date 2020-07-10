!
!  Program to test object composition from Fortran
!
      program main

#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscviewer.h>
      use petscsys
      implicit none

      PetscErrorCode                 ierr
      PetscObject                    o1, o2, o3
      character*(PETSC_MAX_PATH_LEN) name

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
         print*, 'Unable to begin PETSc program'
         stop
      endif

      call PetscViewerCreate(PETSC_COMM_WORLD,o1,ierr);CHKERRA(ierr)
      call PetscViewerCreate(PETSC_COMM_WORLD,o2,ierr);CHKERRA(ierr)
      name = 'matt'
      call PetscObjectCompose(o1,name,o2,ierr);CHKERRA(ierr)
      call PetscObjectQuery(o1,name,o3,ierr);CHKERRA(ierr)
      if (o2 .ne. o3) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'PetscObjectQuery failed'); endif

      call PetscViewerDestroy(o1,ierr);CHKERRA(ierr)
      call PetscViewerDestroy(o2,ierr);CHKERRA(ierr)
      call PetscFinalize(ierr)
      end

!
!/*TEST
!
!   test:
!      suffix: 0
!
!TEST*/
