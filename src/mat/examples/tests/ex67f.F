
C      "$Id: ex58.F,v 1.1 1997/07/24 18:01:48 bsmith Exp bsmith $";
C
C   This program demonstrates use of MatGetRow() from Fortran
C

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/mat.h"
#include "include/FINCLUDE/is.h"
#include "include/FINCLUDE/viewer.h"

      Mat      A,B
      integer  err,i
      Scalar   values(500)
      integer  err,cols(500),ncols
      Viewer   v    
      IS       isrow,iscol

      call PetscInitialize(PETSC_NULL_CHARACTER,err)

      call ViewerFileOpenBinary(PETSC_COMM_WORLD,'small',BINARY_RDONLY,
     &                          v,err)

      call MatLoad(v,MATSEQAIJ,A,err)

!      call MatView(A,VIEWER_STDOUT_WORLD,err)
!
!      call MatGetRow(A,2,ncols,cols,values,err)
!      do 10, i=1,ncols
!        print*,i,cols(i),values(i)
! 10   continue
!      call MatRestoreRow(A,2,ncols,cols,values,err)

!      call MatGetRow(A,5,ncols,cols,values,err)
!      do 20, i=1,ncols
!        print*,i,cols(i),values(i)
! 20   continue
!      call MatRestoreRow(A,2,ncols,cols,values,err)

      call ISCreateGeneral(PETSC_COMM_SELF,
     &               1,0,isrow,ierr)

      call MatGetSubmatrices(A,1,isrow,isrow,
     &        MAT_INITIAL_MATRIX,B,ierr)

      call MatDestroy(A,err)
      call ViewerDestroy(v,err)

      call PetscFinalize(err)
      end




