!
!   This program demonstrates use of MatGetSubMatrices() from Fortran
!
      program main
      implicit none
#include "include/finclude/petsc.h"
#include "include/finclude/petscmat.h"
#include "include/finclude/petscis.h"
#include "include/finclude/petscviewer.h"

      Mat      A,B
      integer  ierr
      PetscViewer   v    
      IS       isrow

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'../matbinary.ex',         &
     &                          PETSC_FILE_RDONLY,v,ierr)

      call MatLoad(v,MATSEQAIJ,A,ierr)

      call ISCreateGeneral(PETSC_COMM_SELF,1,0,isrow,ierr)

      call MatGetSubmatrices(A,1,isrow,isrow,                           &
     &        MAT_INITIAL_MATRIX,B,ierr)

      call MatView(B,PETSC_VIEWER_STDOUT_SELF,ierr)

      call MatGetSubmatrices(A,1,isrow,isrow,                           &
     &        MAT_REUSE_MATRIX,B,ierr)

      call MatView(B,PETSC_VIEWER_STDOUT_SELF,ierr)

      call ISDestroy(isrow,ierr)
      call MatDestroy(A,ierr)
      call MatDestroy(B,ierr)
      call PetscViewerDestroy(v,ierr)

      call PetscFinalize(ierr)
      end




