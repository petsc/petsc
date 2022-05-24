program main
#include <petsc/finclude/petscvec.h>
#include <petsc/finclude/petscmat.h>

use petscvec
use petscmat

implicit none

  Mat             A
  PetscInt,parameter ::  n=5,m=5
  PetscScalar,parameter ::  two =2.0, one = 1.0
  PetscInt,pointer,dimension(:) ::  dnnz,onnz
  PetscInt    ::  i,rstart,rend,M1,N1
  PetscErrorCode  ierr

  PetscCallA(PetscInitialize(ierr))

  allocate(dnnz(0:m-1))
  allocate(onnz(0:m-1))

  do i=0,m-1
   dnnz(i) = 1
   onnz(i) = 1
  end do

  PetscCallA(MatCreateAIJ(PETSC_COMM_WORLD,m,n,PETSC_DETERMINE,PETSC_DETERMINE,PETSC_DECIDE,dnnz,PETSC_DECIDE,onnz,A,ierr))
  PetscCallA(MatSetFromOptions(A,ierr))
  PetscCallA(MatSetUp(A,ierr))
  deallocate(dnnz)
  deallocate(onnz)

  !/* This assembly shrinks memory because we do not insert enough number of values */
  PetscCallA(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr))
  PetscCallA(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr))

  !/* MatResetPreallocation restores the memory required by users */
  PetscCallA(MatResetPreallocation(A,ierr))
  PetscCallA(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE,ierr))
  PetscCallA(MatGetOwnershipRange(A,rstart,rend,ierr))
  PetscCallA(MatGetSize(A,M1,N1,ierr))
  do i=rstart,rend-1
   PetscCallA(MatSetValue(A,i,i,two,INSERT_VALUES,ierr))
   if (rend<N1) PetscCallA(MatSetValue(A,i,rend,one,INSERT_VALUES,ierr))
  end do
  PetscCallA(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr))
  PetscCallA(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr))
  PetscCallA(MatView(A,PETSC_VIEWER_STDOUT_WORLD,ierr))
  PetscCallA(MatDestroy(A,ierr))
  PetscCallA(PetscFinalize(ierr))

end program

!/*TEST
!
!   test:
!      suffix: 1
!      output_file: output/ex4_1.out
!
!   test:
!      suffix: 2
!      nsize: 2
!      output_file: output/ex4_2.out
!
!TEST*/
