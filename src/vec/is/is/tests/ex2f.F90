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
      PetscCheckA(n .eq. 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,'Wrong result from ISCreateStride')

      PetscCallA(ISStrideGetInfo(is,start,stride,ierr))
      PetscCheckA(start .eq. 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,'Wrong result from ISStrideGetInfo')
      PetscCheckA(stride .eq. 2,PETSC_COMM_SELF,PETSC_ERR_PLIB,'Wrong result from ISStrideGetInfo')

      PetscCallA(PetscObjectTypeCompare(is,ISSTRIDE,flag,ierr))
      PetscCheckA(flag,PETSC_COMM_SELF,PETSC_ERR_PLIB,'Wrong result from PetscObjectTypeCompare')
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
        PetscCheckA(ii(i) .eq. -11 + 3*i,PETSC_COMM_SELF,PETSC_ERR_PLIB,'Wrong result from ISGetIndices')
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
