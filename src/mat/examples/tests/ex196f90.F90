!
!
!   This program demonstrates use of MatSeqAIJGetArrayF90()
!
      program main
#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      Mat      A
      PetscErrorCode ierr
      PetscViewer   v
      PetscScalar, pointer :: aa(:)
      character*(256)  f
      PetscBool flg

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif

      call PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-f',f,flg,ierr);CHKERRA(ierr)
      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,f,FILE_MODE_READ,v,ierr);CHKERRA(ierr)

      call MatCreate(PETSC_COMM_WORLD,A,ierr);CHKERRA(ierr)
      call MatSetType(A, MATSEQAIJ,ierr);CHKERRA(ierr)
      call MatLoad(A,v,ierr);CHKERRA(ierr)

      call MatView(A,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)

      call MatSeqAIJGetArrayF90(A,aa,ierr);CHKERRA(ierr)
      print*,aa(3)

      call MatDestroy(A,ierr);CHKERRA(ierr)
      call PetscViewerDestroy(v,ierr);CHKERRA(ierr)

      call PetscFinalize(ierr)
      end




