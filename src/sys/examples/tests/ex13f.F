!
!  Progran to test object composition
!
      program main

#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscviewer.h>
      use petscsys
      implicit none

      PetscErrorCode ierr
      PetscObject    o1, o2, o3
      character*(80) name

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
         print*, 'Unable to begin PETSc program'
      endif

      call PetscViewerCreate(PETSC_COMM_WORLD,o1,ierr)
      CHKERRQ(ierr)
      call PetscViewerCreate(PETSC_COMM_WORLD,o2,ierr)
      CHKERRQ(ierr)
      name = 'matt'
      call PetscObjectCompose(o1,name,o2,ierr)
      CHKERRQ(ierr)
      call PetscObjectQuery(o1,name,o3,ierr)
      CHKERRQ(ierr)
      if (o2 .ne. o3) then
         CHKERRQ(PETSC_ERR_PLIB)
      endif

      call PetscViewerDestroy(o1,ierr)
      CHKERRQ(ierr)
      call PetscViewerDestroy(o2,ierr)
      CHKERRQ(ierr)
      call PetscFinalize(ierr)
      end

!
!/*TEST
!
!   test:
!      suffix: 0
!
!TEST*/
