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

  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

  if (ierr /= 0) then
   print*,'PetscInitialize failed'
   stop
  endif

  allocate(dnnz(0:m-1))
  allocate(onnz(0:m-1))

  do i=0,m-1
   dnnz(i) = 1
   onnz(i) = 1
  end do

  call MatCreateAIJ(PETSC_COMM_WORLD,m,n,PETSC_DETERMINE,PETSC_DETERMINE,PETSC_DECIDE,dnnz,PETSC_DECIDE,onnz,A,ierr);CHKERRA(ierr)
  call MatSetFromOptions(A,ierr);CHKERRA(ierr)
  call MatSetUp(A,ierr);CHKERRA(ierr)
  deallocate(dnnz)
  deallocate(onnz)

  !/* This assembly shrinks memory because we do not insert enough number of values */
  call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr);CHKERRA(ierr)
  call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr);CHKERRA(ierr)

  !/* MatResetPreallocation restores the memory required by users */
  call MatResetPreallocation(A,ierr);CHKERRA(ierr)
  call MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE,ierr);CHKERRA(ierr)
  call MatGetOwnershipRange(A,rstart,rend,ierr);CHKERRA(ierr)
  call MatGetSize(A,M1,N1,ierr);CHKERRA(ierr)
  do i=rstart,rend-1
   call MatSetValue(A,i,i,two,INSERT_VALUES,ierr);CHKERRA(ierr)
   if (rend<N1) call MatSetValue(A,i,rend,one,INSERT_VALUES,ierr);CHKERRA(ierr)
  end do
  call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr);CHKERRA(ierr)
  call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr);CHKERRA(ierr)
  call MatView(A,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)
  call MatDestroy(A,ierr);CHKERRA(ierr)
  call PetscFinalize(ierr);CHKERRA(ierr)

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
