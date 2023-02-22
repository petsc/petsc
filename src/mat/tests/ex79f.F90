!
!   This program demonstrates use of MatGetRowIJ() from Fortran
!
      program main

#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      Mat         A,Ad,Ao
      PetscErrorCode ierr
      PetscMPIInt rank
      PetscViewer v
      PetscInt i,j
      PetscInt n,rstart
      PetscInt zero,one,rend
      PetscBool   done,bb
      PetscScalar,pointer :: aa(:)
      PetscInt,pointer :: ia(:),ja(:),icol(:)

      PetscCallA(PetscInitialize(ierr))
      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))

      PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_WORLD,'${PETSC_DIR}/share/petsc/datafiles/matrices/' // 'ns-real-int32-float64',FILE_MODE_READ,v,ierr))
      PetscCallA(MatCreate(PETSC_COMM_WORLD,A,ierr))
      PetscCallA(MatSetType(A, MATMPIAIJ,ierr))
      PetscCallA(MatLoad(A,v,ierr))
      PetscCallA(MatView(A,PETSC_VIEWER_STDOUT_WORLD,ierr))

      PetscCallA(MatMPIAIJGetSeqAIJF90(A,Ad,Ao,icol,ierr))
      PetscCallA(MatGetOwnershipRange(A,rstart,rend,ierr))
!
!   Print diagonal portion of matrix
!
      PetscCallA(PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1,ierr))
      zero = 0
      one  = 1
      bb = .true.
      PetscCallA(MatGetRowIJF90(Ad,one,bb,bb,n,ia,ja,done,ierr))
      PetscCallA(MatSeqAIJGetArrayF90(Ad,aa,ierr))
      do 10, i=1,n
        write(7+rank,*) 'row ',i+rstart,' number nonzeros ',ia(i+1)-ia(i)
        do 20, j=ia(i),ia(i+1)-1
          write(7+rank,*)'  ',j,ja(j)+rstart,aa(j)
 20     continue
 10   continue
      PetscCallA(MatRestoreRowIJF90(Ad,one,bb,bb,n,ia,ja,done,ierr))
      PetscCallA(MatSeqAIJRestoreArrayF90(Ad,aa,ierr))
!
!   Print off-diagonal portion of matrix
!
      PetscCallA(MatGetRowIJF90(Ao,one,bb,bb,n,ia,ja,done,ierr))
      PetscCallA(MatSeqAIJGetArrayF90(Ao,aa,ierr))
      do 30, i=1,n
        write(7+rank,*) 'row ',i+rstart,' number nonzeros ',ia(i+1)-ia(i)
        do 40, j=ia(i),ia(i+1)-1
          write(7+rank,*)'  ',j,icol(ja(j))+1,aa(j)
 40     continue
 30   continue
      PetscCallA(MatMPIAIJRestoreSeqAIJF90(A,Ad,Ao,icol,ierr))
      PetscCallA(MatRestoreRowIJF90(Ao,one,bb,bb,n,ia,ja,done,ierr))
      PetscCallA(MatSeqAIJRestoreArrayF90(Ao,aa,ierr))

      PetscCallA(PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1,ierr))

      PetscCallA(MatGetDiagonalBlock(A,Ad,ierr))
      PetscCallA(MatView(Ad,PETSC_VIEWER_STDOUT_WORLD,ierr))

      PetscCallA(MatView(A,PETSC_VIEWER_STDOUT_WORLD,ierr))
      PetscCallA(MatDestroy(A,ierr))
      PetscCallA(PetscViewerDestroy(v,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!     build:
!       requires: double !complex !defined(PETSC_USE_64BIT_INDICES)
!
!     test:
!        args: -binary_read_double -options_left false
!
!TEST*/
