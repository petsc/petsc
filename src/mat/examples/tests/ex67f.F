
C      "$Id: ex67f.F,v 1.4 1999/02/18 16:37:46 balay Exp bsmith $";
C
C   This program demonstrates use of MatGetSubMatrices() from Fortran
C

#include "include/finclude/petsc.h"
#include "include/finclude/mat.h"
#include "include/finclude/is.h"
#include "include/finclude/viewer.h"

      Mat      A,B
      integer  ierr
      Viewer   v    
      IS       isrow

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      call ViewerBinaryOpen(PETSC_COMM_WORLD,'../matbinary.ex',         &
     &                          BINARY_RDONLY,v,ierr)

      call MatLoad(v,MATSEQAIJ,A,ierr)

      call ISCreateGeneral(PETSC_COMM_SELF,1,0,isrow,ierr)

      call MatGetSubmatrices(A,1,isrow,isrow,                           &
     &        MAT_INITIAL_MATRIX,B,ierr)

      call MatView(B,VIEWER_STDOUT_SELF,ierr)

      call MatGetSubmatrices(A,1,isrow,isrow,                           &
     &        MAT_REUSE_MATRIX,B,ierr)

      call MatView(B,VIEWER_STDOUT_SELF,ierr)

      call ISDestroy(isrow,ierr)
      call MatDestroy(A,ierr)
      call MatDestroy(B,ierr)
      call ViewerDestroy(v,ierr)

      call PetscFinalize(ierr)
      end




