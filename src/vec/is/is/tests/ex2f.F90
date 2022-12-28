!
!  Formatted Test for IS stride routines
!
      program main
#include <petsc/finclude/petscis.h>
      use petscis
      implicit none

      PetscErrorCode ierr
      PetscInt  i,n,start
      PetscInt  stride,ssize,first
      IS          is
      PetscBool   flag
      PetscInt, pointer :: ii(:)

      PetscCallA(PetscInitialize(ierr))

!     Test IS of size 0
      ssize = 0
      stride = 0
      first = 2
      PetscCallA(ISCreateStride(PETSC_COMM_SELF,ssize,stride,first,is,ierr))
      PetscCallA(ISGetLocalSize(is,n,ierr))
      if (n .ne. 0) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Wrong result from ISCreateStride'); endif

      PetscCallA(ISStrideGetInfo(is,start,stride,ierr))
      if (start .ne. 0) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Wrong result from ISStrideGetInfo'); endif

      if (stride .ne. 2) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Wrong result from ISStrideGetInfo') ; endif

      PetscCallA(PetscObjectTypeCompare(is,ISSTRIDE,flag,ierr))
      if (.not. flag) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Wrong result from PetscObjectTypeCompare'); endif
      PetscCallA(ISGetIndicesF90(is,ii,ierr))
      PetscCallA(ISRestoreIndicesF90(is,ii,ierr))
      PetscCallA(ISDestroy(is,ierr))

!     Test ISGetIndices()

      ssize = 10000
      stride = -8
      first = 3
      PetscCallA(ISCreateStride(PETSC_COMM_SELF,ssize,stride,first,is,ierr))
      PetscCallA(ISGetLocalSize(is,n,ierr))
      PetscCallA(ISGetIndicesF90(is,ii,ierr))
      do 10, i=1,n
        if (ii(i) .ne. -11 + 3*i) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Wrong result from ISGetIndices'); endif
 10   continue
      PetscCallA(ISRestoreIndicesF90(is,ii,ierr))
      PetscCallA(ISDestroy(is,ierr))

      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   test:
!     output_file: output/ex1_1.out
!
!TEST*/
