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

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif

      call MatCreate(PETSC_COMM_WORLD,A,ierr);CHKERRA(ierr)
      three = 3
      call MatSetSizes(A,three,three,three,three,ierr);CHKERRA(ierr)
      call MatSetBlockSize(A,three,ierr);CHKERRA(ierr)
      call MatSetType(A, MATSEQBAIJ,ierr);CHKERRA(ierr)
      call MatSetUp(A,ierr);CHKERRA(ierr)

      one = 1
      idxm(1) = 0
      allocate (km(three,three))
      do i=1,3
        do j=1,3
          km(i,j) = i + j
        enddo
      enddo

      call MatSetValuesBlocked(A, one, idxm, one, idxm, km, ADD_VALUES, ierr);CHKERRA(ierr)
      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr);CHKERRA(ierr)
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr);CHKERRA(ierr)
      call MatView(A,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)

      j = 0
      call MatGetValues(A,one,j,one,j,v,ierr);CHKERRA(ierr)

      call MatDestroy(A,ierr);CHKERRA(ierr)

      deallocate(km)
      call PetscFinalize(ierr)
      end

!/*TEST
!
!     test:
!       requires: double !complex
!
!TEST*/
