!
!     Test for bug with ISGetIndicesF90() when length of indices is 0
!
!     Contributed by: Jakub Fabian
!
program main
#include <petsc/finclude/petscis.h>
  use petscis
  implicit none

  PetscErrorCode ierr
  PetscInt n, bs
  PetscInt, pointer :: indices(:)=>NULL()
  PetscInt, pointer :: idx(:)=>NULL()
  IS      is

  n = 0
  allocate(indices(n), source=n)

  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

  call ISCreateGeneral(PETSC_COMM_SELF,n,indices,PETSC_USE_POINTER,is,ierr);CHKERRA(ierr)
  call ISGetIndicesF90(is,idx,ierr);CHKERRA(ierr)
  call ISRestoreIndicesF90(is,idx,ierr);CHKERRA(ierr)
  call ISDestroy(is,ierr);CHKERRA(ierr)

  bs = 2
  call ISCreateBlock(PETSC_COMM_SELF,bs,n,indices,PETSC_USE_POINTER,is,ierr);CHKERRA(ierr)
  call ISGetIndicesF90(is,idx,ierr);CHKERRA(ierr)
  call ISRestoreIndicesF90(is,idx,ierr);CHKERRA(ierr)
  call ISDestroy(is,ierr);CHKERRA(ierr)
  call PetscFinalize(ierr)
end

!/*TEST
!
!   test:
!
!TEST*/
