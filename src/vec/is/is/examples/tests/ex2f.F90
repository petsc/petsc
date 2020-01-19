!
!  Formatted Test for IS stride routines
!
      program main
#include <petsc/finclude/petscis.h>
      use petscis
      implicit none

      PetscErrorCode ierr
      PetscInt  i,n,ii(1),start
      PetscInt  stride,ssize,first
      IS          is
      PetscBool   flag
      PetscOffset iis

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      
!     Test IS of size 0
      ssize = 0
      stride = 0
      first = 2
      call ISCreateStride(PETSC_COMM_SELF,ssize,stride,first,is,ierr)
      call ISGetLocalSize(is,n,ierr)
      if (n .ne. 0) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Wrong result from ISCreateStride'); endif

      call ISStrideGetInfo(is,start,stride,ierr)
      if (start .ne. 0) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Wrong result from ISStrideGetInfo'); endif

      if (stride .ne. 2) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Wrong result from ISStrideGetInfo') ; endif

      call PetscObjectTypeCompare(is,ISSTRIDE,flag,ierr)
      if (.not. flag) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Wrong result from PetscObjectTypeCompare'); endif
      call ISGetIndices(is,ii,iis,ierr)
      call ISRestoreIndices(is,ii,iis,ierr)
      call ISDestroy(is,ierr)

!     Test ISGetIndices()

      ssize = 10000
      stride = -8
      first = 3
      call ISCreateStride(PETSC_COMM_SELF,ssize,stride,first,is,ierr)
      call ISGetLocalSize(is,n,ierr)
      call ISGetIndices(is,ii,iis,ierr)
      do 10, i=1,n
        if (ii(i+iis) .ne. -11 + 3*i) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Wrong result from ISGetIndices'); endif
 10   continue
      call ISRestoreIndices(is,ii,iis,ierr)
      call ISDestroy(is,ierr)

      call PetscFinalize(ierr)
      end

!/*TEST
!
!   test:
!     output_file: output/ex1_1.out
!
!TEST*/
