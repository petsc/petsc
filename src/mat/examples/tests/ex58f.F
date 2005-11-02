!
!
!   This program demonstrates use of MatGetRow() from Fortran
!
      program main
#include "include/finclude/petsc.h"
#include "include/finclude/petscmat.h"
#include "include/finclude/petscviewer.h"

      Mat      A
      PetscErrorCode ierr
      PetscInt i,cols(500),ncols,row
      PetscScalar   values(500)
      PetscViewer   v    

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'small',              &
     & FILE_MODE_READ,v,ierr)

      call MatLoad(v,MATSEQAIJ,A,ierr)

      call MatView(A,PETSC_VIEWER_STDOUT_WORLD,ierr)

      row = 2
      call MatGetRow(A,row,ncols,cols,values,ierr)
      do 10, i=1,ncols
        print*,i,cols(i),values(i)
 10   continue
      call MatRestoreRow(A,row,ncols,cols,values,ierr)

      row = 5
      call MatGetRow(A,row,ncols,cols,values,ierr)
      do 20, i=1,ncols
        print*,i,cols(i),values(i)
 20   continue
      call MatRestoreRow(A,row,ncols,cols,values,ierr)

      call MatDestroy(A,ierr)
      call PetscViewerDestroy(v,ierr)

      call PetscFinalize(ierr)
      end




