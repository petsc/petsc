!
!
!
      program main
#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      Mat      A
      PetscErrorCode ierr
      PetscScalar, pointer :: km(:,:)
      PetscInt three,one
      PetscInt idxm(1),i,j
      PetscScalar v

      PetscCallA(PetscInitialize(ierr))

      PetscCallA(MatCreate(PETSC_COMM_WORLD,A,ierr))
      three = 3
      PetscCallA(MatSetSizes(A,three,three,three,three,ierr))
      PetscCallA(MatSetBlockSize(A,three,ierr))
      PetscCallA(MatSetType(A, MATSEQBAIJ,ierr))
      PetscCallA(MatSetUp(A,ierr))

      one = 1
      idxm(1) = 0
      allocate (km(three,three))
      do i=1,3
        do j=1,3
          km(i,j) = i + j
        enddo
      enddo

      PetscCallA(MatSetValuesBlocked(A, one, idxm, one, idxm, km, ADD_VALUES, ierr))
      PetscCallA(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr))
      PetscCallA(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr))
      PetscCallA(MatView(A,PETSC_VIEWER_STDOUT_WORLD,ierr))

      j = 0
      PetscCallA(MatGetValues(A,one,j,one,j,v,ierr))

      PetscCallA(MatDestroy(A,ierr))

      deallocate(km)
      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!     test:
!       requires: double !complex
!
!TEST*/
