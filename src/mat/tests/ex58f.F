!
!
!   This program demonstrates use of MatGetRow() and MatGetRowMaxAbs() from Fortran
!
      program main
#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      Mat      A
      PetscErrorCode ierr
      PetscInt M,N
      PetscViewer   v
      Vec           rowmax
      PetscBool flg
      character*(256)  f

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif

      call PetscOptionsGetString(PETSC_NULL_OPTIONS,                           &
     &      PETSC_NULL_CHARACTER,'-f',f,flg,ierr)
      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,f,FILE_MODE_READ,v,          &
     &                           ierr)

      call MatCreate(PETSC_COMM_WORLD,A,ierr)
      call MatSetType(A, MATSEQAIJ,ierr)
      call MatLoad(A,v,ierr)

      call MatView(A,PETSC_VIEWER_STDOUT_WORLD,ierr)

!
!     Test MatGetRowMaxAbs()
      call MatGetSize(A,M,N,ierr)
      call VecCreate(PETSC_COMM_WORLD,rowmax,ierr)
      call VecSetSizes(rowmax,M,M,ierr)
      call VecSetFromOptions(rowmax,ierr)

      call MatGetRowMaxAbs(A,rowmax,PETSC_NULL_INTEGER,ierr)
      call VecView(rowmax,PETSC_VIEWER_STDOUT_WORLD,ierr)

      call MatGetRowMax(A,rowmax,PETSC_NULL_INTEGER,ierr)
      call VecView(rowmax,PETSC_VIEWER_STDOUT_WORLD,ierr)

      call MatGetRowMinAbs(A,rowmax,PETSC_NULL_INTEGER,ierr)
      call VecView(rowmax,PETSC_VIEWER_STDOUT_WORLD,ierr)

      call MatGetRowMin(A,rowmax,PETSC_NULL_INTEGER,ierr)
      call VecView(rowmax,PETSC_VIEWER_STDOUT_WORLD,ierr)

      call MatDestroy(A,ierr)
      call PetscViewerDestroy(v,ierr)
      call VecDestroy(rowmax,ierr)

      call PetscFinalize(ierr)
      end

!/*TEST
!
!     test:
!       args: -f ${DATAFILESPATH}/matrices/tiny
!       requires: datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
!
!TEST*/
