!
!
!   This program tests storage of PETSc Dense matrix.
!   It Creates a Dense matrix, and stores it into a file,
!   and then reads it back in as a SeqDense and MPIDense
!   matrix, and prints out the contents.
!
      program main
#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      PetscErrorCode ierr
      PetscInt row,col,ten
      PetscMPIInt rank
      PetscScalar  v
      Mat     A
      PetscViewer  view

      PetscCallA(PetscInitialize(ierr))

      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))
!
!     Proc-0 Create a seq-dense matrix and write it to a file
!
      if (rank .eq. 0) then
         ten = 10
         PetscCallA(MatCreateSeqDense(PETSC_COMM_SELF,ten,ten,PETSC_NULL_SCALAR,A,ierr))
         v = 1.0
         do row=0,9
            do col=0,9
               PetscCallA(MatSetValue(A,row,col,v,INSERT_VALUES,ierr))
               v = v + 1.0
            enddo
         enddo

         PetscCallA(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr))
         PetscCallA(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr))

         PetscCallA(PetscObjectSetName(A,'Original Matrix',ierr))
         PetscCallA(MatView(A,PETSC_VIEWER_STDOUT_SELF,ierr))
!
!        Now Write this matrix to a binary file
!
         PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_SELF,'dense.mat',FILE_MODE_WRITE,view,ierr))
         PetscCallA(MatView(A,view,ierr))
         PetscCallA(PetscViewerDestroy(view,ierr))
         PetscCallA(MatDestroy(A,ierr))
!
!        Read this matrix into a SeqDense matrix

         PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_SELF,'dense.mat',FILE_MODE_READ,view,ierr))
         PetscCallA(MatCreate(PETSC_COMM_SELF,A,ierr))
         PetscCallA(MatSetType(A, MATSEQDENSE,ierr))
         PetscCallA(MatLoad(A,view,ierr))

         PetscCallA(PetscObjectSetName(A,'SeqDense Matrix read in from file',ierr))
         PetscCallA(MatView(A,PETSC_VIEWER_STDOUT_SELF,ierr))
         PetscCallA(MatDestroy(A,ierr))
         PetscCallA(PetscViewerDestroy(view,ierr))
      endif

!
!     Use a barrier, so that the procs do not try opening the file before
!     it is created.
!
      PetscCallMPIA(MPI_Barrier(PETSC_COMM_WORLD,ierr))

      PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_WORLD,'dense.mat',FILE_MODE_READ,view,ierr))
      PetscCallA(MatCreate(PETSC_COMM_WORLD,A,ierr))
      PetscCallA(MatSetType(A, MATMPIDENSE,ierr))
      PetscCallA(MatLoad(A,view,ierr))

      PetscCallA(PetscObjectSetName(A, 'MPIDense Matrix read in from file',ierr))
      PetscCallA(MatView(A,PETSC_VIEWER_STDOUT_WORLD,ierr))
      PetscCallA(MatDestroy(A,ierr))
      PetscCallA(PetscViewerDestroy(view,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   test:
!      nsize: 2
!      output_file: output/ex63_1.out
!
!TEST*/
