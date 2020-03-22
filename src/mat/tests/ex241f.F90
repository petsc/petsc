!     Test code contributed by Thibaut Appel <t.appel17@imperial.ac.uk>

  program test_assembly

#include <petsc/finclude/petscmat.h>

  use PetscMat
  use ISO_Fortran_Env, only : output_unit, real64

  implicit none
  PetscInt,    parameter :: wp = real64, n = 10
  PetscScalar, parameter :: zero = 0.0, one = 1.0
  Mat      :: L
  PetscInt :: istart, iend, row, i1, i0
  PetscErrorCode :: ierr

  PetscInt    cols(1),rows(1)
  PetscScalar  vals(1)

  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
  if (ierr .ne. 0) then
    print*,'Unable to initialize PETSc'
    stop
  endif

  i0 = 0
  i1 = 1

  call MatCreate(PETSC_COMM_WORLD,L,ierr); CHKERRA(ierr)
  call MatSetType(L,MATAIJ,ierr); CHKERRA(ierr)
  call MatSetSizes(L,PETSC_DECIDE,PETSC_DECIDE,n,n,ierr); CHKERRA(ierr)

  call MatSeqAIJSetPreallocation(L,i1,PETSC_NULL_INTEGER,ierr); CHKERRA(ierr)
  call MatMPIAIJSetPreallocation(L,i1,PETSC_NULL_INTEGER,i0,PETSC_NULL_INTEGER,ierr); CHKERRA(ierr) ! No allocated non-zero in off-diagonal part
  call MatSetOption(L,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE,ierr); CHKERRA(ierr)
  call MatSetOption(L,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE,ierr); CHKERRA(ierr)
  call MatSetOption(L,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE,ierr); CHKERRA(ierr)

  call MatGetOwnershipRange(L,istart,iend,ierr); CHKERRA(ierr)

  ! assembling a diagonal matrix
  do row = istart,iend-1

    cols = [row]; vals = [one]; rows = [row];
    call MatSetValues(L,i1,rows,i1,cols,vals,ADD_VALUES,ierr); CHKERRA(ierr)

  end do

  call MatAssemblyBegin(L,MAT_FINAL_ASSEMBLY,ierr); CHKERRA(ierr)
  call MatAssemblyEnd(L,MAT_FINAL_ASSEMBLY,ierr); CHKERRA(ierr)

  call MatSetOption(L,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE,ierr); CHKERRA(ierr)

  !call MatZeroEntries(L,ierr); CHKERRA(ierr)

  ! assembling a diagonal matrix, adding a zero value to non-diagonal part
  do row = istart,iend-1

    if (row == 0) then
      cols = [n-1]
      vals = [zero]
      rows = [row]
      call MatSetValues(L,i1,rows,i1,cols,vals,ADD_VALUES,ierr); CHKERRA(ierr)
    end if
    cols = [row]; vals = [one] ; rows = [ row];
    call MatSetValues(L,i1,rows,i1,cols,vals,ADD_VALUES,ierr); CHKERRA(ierr)

  end do

  call MatAssemblyBegin(L,MAT_FINAL_ASSEMBLY,ierr); CHKERRA(ierr)
  call MatAssemblyEnd(L,MAT_FINAL_ASSEMBLY,ierr); CHKERRA(ierr)
  call MatDestroy(L,ierr); CHKERRA(ierr)

  call PetscFinalize(ierr)

end program test_assembly

!/*TEST
!
!   build:
!      requires: complex
!
!   test:
!      nsize: 2
!
!TEST*/
