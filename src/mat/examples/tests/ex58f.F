!
!      "$Id: ex58f.F,v 1.7 1999/03/26 23:20:06 balay Exp bsmith $";
!
!   This program demonstrates use of MatGetRow() from Fortran
!

#include "include/finclude/petsc.h"
#include "include/finclude/mat.h"
#include "include/finclude/viewer.h"

      Mat      A
      integer  err,i,cols(500),ncols
      Scalar   values(500)
      Viewer   v    

      call PetscInitialize(PETSC_NULL_CHARACTER,err)

      call ViewerBinaryOpen(PETSC_COMM_WORLD,'small',BINARY_RDONLY,     &
     &                          v,err)

      call MatLoad(v,MATSEQAIJ,A,err)

      call MatView(A,VIEWER_STDOUT_WORLD,err)

      call MatGetRow(A,2,ncols,cols,values,err)
      do 10, i=1,ncols
        print*,i,cols(i),values(i)
 10   continue
      call MatRestoreRow(A,2,ncols,cols,values,err)

      call MatGetRow(A,5,ncols,cols,values,err)
      do 20, i=1,ncols
        print*,i,cols(i),values(i)
 20   continue
      call MatRestoreRow(A,2,ncols,cols,values,err)

      call MatDestroy(A,err)
      call ViewerDestroy(v,err)

      call PetscFinalize(err)
      end




