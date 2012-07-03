!
!
!   This program demonstrates use of MatGetRow() and MatGetRowMaxAbs() from Fortran
!
      program main
#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscmat.h>
#include <finclude/petscviewer.h>

      Mat      A
      PetscErrorCode ierr
      PetscInt i,cols(500),ncols,row,M,N
      PetscScalar   values(500)
      PetscViewer   v    
      Vec           rowmax

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'small',              &
     & FILE_MODE_READ,v,ierr)

      call MatCreate(PETSC_COMM_WORLD,A,ierr)
      call MatSetType(A, MATSEQAIJ,ierr)
      call MatLoad(A,v,ierr)

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

!
!     Test MatGetRowMaxAbs()
      call MatGetSize(A,M,N,ierr)
      call VecCreate(PETSC_COMM_WORLD,rowmax,ierr)
      call VecSetSizes(rowmax,M,M,ierr)
      call VecSetFromOptions(rowmax,ierr)

      call MatGetRowMaxAbs(A,rowmax,PETSC_NULL_INTEGER,ierr)
      call VecView(rowmax,PETSC_VIEWER_STDOUT_WORLD,ierr)

      call MatDestroy(A,ierr)
      call PetscViewerDestroy(v,ierr)
      call VecDestroy(rowmax,ierr)

      call PetscFinalize(ierr)
      end




