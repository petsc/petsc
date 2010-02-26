!
!   This program demonstrates use of MatGetSubMatrices() from Fortran
!
      program main
      implicit none
#include "finclude/petscsys.h"
#include "finclude/petscmat.h"
#include "finclude/petscis.h"
#include "finclude/petscviewer.h"

      Mat      A,B
      PetscErrorCode  ierr
      PetscInt one,zero
      PetscViewer   v    
      IS       isrow

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'../matbinary.ex',         &
     &                          FILE_MODE_READ,v,ierr)

      call MatLoad(v,MATSEQAIJ,A,ierr)

      one  = 1
      zero = 0
      call ISCreateGeneral(PETSC_COMM_SELF,one,zero,isrow,ierr)

      call MatGetSubmatrices(A,one,isrow,isrow,                           &
     &        MAT_INITIAL_MATRIX,B,ierr)

      call MatView(B,PETSC_VIEWER_STDOUT_SELF,ierr)

      call MatGetSubmatrices(A,one,isrow,isrow,                           &
     &        MAT_REUSE_MATRIX,B,ierr)

      call MatView(B,PETSC_VIEWER_STDOUT_SELF,ierr)

      call ISDestroy(isrow,ierr)
      call MatDestroy(A,ierr)
      call MatDestroy(B,ierr)
      call PetscViewerDestroy(v,ierr)

      call PetscFinalize(ierr)
      end




