!
!
!   This program demonstrates use of MatGetRow() from Fortran
!
      program main
#include "include/finclude/petsc.h"
#include "include/finclude/petscmat.h"
#include "include/finclude/petscviewer.h"

      Mat      A
      integer  err,i,cols(500),ncols
      PetscScalar   values(500)
      PetscViewer   v    

      call PetscInitialize(PETSC_NULL_CHARACTER,err)

      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'small',              &
     & PETSC_FILE_RDONLY,v,err)

      call MatLoad(v,MATSEQAIJ,A,err)

      call MatView(A,PETSC_VIEWER_STDOUT_WORLD,err)

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
      call PetscViewerDestroy(v,err)

      call PetscFinalize(err)
      end




