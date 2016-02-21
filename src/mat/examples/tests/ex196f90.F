!
!
!   This program demonstrates use of MatSeqAIJGetArrayF90()
!
      program main
      implicit none
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscvec.h>
#include <petsc/finclude/petscmat.h>
#include <petsc/finclude/petscviewer.h>
#include <petsc/finclude/petscmat.h90>

      Mat      A
      PetscErrorCode ierr
      PetscViewer   v
      PetscScalar, pointer :: aa(:)
      character*(256)  f
      PetscBool flg

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      call PetscOptionsGetString(PETSC_NULL_OBJECT,PETSC_NULL_CHARACTER,       &
     &                           '-f',f,flg,ierr)
      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,f,FILE_MODE_READ,v,          &
     &                           ierr)

      call MatCreate(PETSC_COMM_WORLD,A,ierr)
      call MatSetType(A, MATSEQAIJ,ierr)
      call MatLoad(A,v,ierr)

      call MatView(A,PETSC_VIEWER_STDOUT_WORLD,ierr)

      call MatSeqAIJGetArrayF90(A,aa,ierr)
      print*,aa(3)

      call MatDestroy(A,ierr)
      call PetscViewerDestroy(v,ierr)

      call PetscFinalize(ierr)
      end




