!
!      "$Id: ex36.F,v 1.25 1998/04/21 18:30:37 balay Exp $";
!
!   This program tests storage of PETSc Dense matrix. 
      implicit none
#include "include/finclude/petsc.h"
#include "include/finclude/mat.h"
#include "include/finclude/vec.h"
#include "include/finclude/viewer.h"
      integer ierr,row,col,rank
      Scalar  v
      Mat     A
      Viewer  view

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
!
!     Proc-0 Create a seq-dense matrix and write it to a file
!
      if (rank .eq. 0) then
         call MatCreateSeqDense(PETSC_COMM_SELF,10,10,                  &
     &        PETSC_NULL_SCALAR,A,ierr)
         v = 1.0d0
         do row=0,9 
            do col=0,9 
               call MatSetValue(A,row,col,v,INSERT_VALUES,ierr)
               v = v + 1.0d0
            enddo
         enddo
         
         call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
         call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)
         write (6,*) 'The Contents of the Original Matrix'
         call MatView(A,VIEWER_STDOUT_SELF,ierr)
!         
!        Now Write this matrix to a binary file
!
         call ViewerFileOpenBinary(PETSC_COMM_SELF,'dense.mat',
     &        BINARY_CREATE,view,ierr)
         call ViewerSetFormat(view,VIEWER_FORMAT_BINARY_NATIVE,         &
     &        PETSC_NULL_CHARACTER,ierr)
         call MatView(A,view,ierr)
         call ViewerDestroy(view,ierr)
         call MatDestroy(A,ierr)
!
!        Read this matrix into a SeqDense matrix

         call ViewerFileOpenBinary(PETSC_COMM_SELF,'dense.mat',
     &        BINARY_RDONLY,view,ierr)
         call MatLoad(view,MATSEQDENSE,A,ierr)
         write (6,*) 'The Contents of SeqDense Matrix read in from file'
         call MatView(A,VIEWER_STDOUT_SELF,ierr)
         call MatDestroy(A,ierr)
         call ViewerDestroy(view,ierr)
      endif

!
!     Use a barrier, so that the procs do not try opening the file before
!     it is created.
!
      call MPI_Barrier(PETSC_COMM_WORLD,ierr)

      call ViewerFileOpenBinary(PETSC_COMM_WORLD,'dense.mat',
     &     BINARY_RDONLY,view,ierr)
      call MatLoad(view,MATMPIDENSE,A,ierr)
      if (rank .eq. 0) then
         write (6,*) 'The Contents of MPIDense Matrix read in from file'
      endif
      call MatView(A,VIEWER_STDOUT_WORLD,ierr)
      call MatDestroy(A,ierr)
      call ViewerDestroy(view,ierr)
      call PetscFinalize(ierr)
      end
