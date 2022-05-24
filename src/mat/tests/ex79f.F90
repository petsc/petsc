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
      PetscInt i,j,ia(1),ja(1)
      PetscInt n,icol(1),rstart
      PetscInt zero,one,rend
      PetscBool   done
      PetscOffset iia,jja,aaa,iicol
      PetscScalar aa(1)

      PetscCallA(PetscInitialize(ierr))
      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))

      PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_WORLD,'${PETSC_DIR}/share/petsc/datafiles/matrices/' // 'ns-real-int32-float64',FILE_MODE_READ,v,ierr))
      PetscCallA(MatCreate(PETSC_COMM_WORLD,A,ierr))
      PetscCallA(MatSetType(A, MATMPIAIJ,ierr))
      PetscCallA(MatLoad(A,v,ierr))
      PetscCallA(MatView(A,PETSC_VIEWER_STDOUT_WORLD,ierr))

      PetscCallA(MatMPIAIJGetSeqAIJ(A,Ad,Ao,icol,iicol,ierr))
      PetscCallA(MatGetOwnershipRange(A,rstart,rend,ierr))
!
!   Print diagonal portion of matrix
!
      PetscCallA(PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1,ierr))
      zero = 0
      one  = 1
      PetscCallA(MatGetRowIJ(Ad,one,zero,zero,n,ia,iia,ja,jja,done,ierr))
      PetscCallA(MatSeqAIJGetArray(Ad,aa,aaa,ierr))
      do 10, i=1,n
        write(7+rank,*) 'row ',i+rstart,' number nonzeros ',ia(iia+i+1)-ia(iia+i)
        do 20, j=ia(iia+i),ia(iia+i+1)-1
          write(7+rank,*)'  ',j,ja(jja+j)+rstart,aa(aaa+j)
 20     continue
 10   continue
      PetscCallA(MatRestoreRowIJ(Ad,one,zero,zero,n,ia,iia,ja,jja,done,ierr))
      PetscCallA(MatSeqAIJRestoreArray(Ad,aa,aaa,ierr))
!
!   Print off-diagonal portion of matrix
!
      PetscCallA(MatGetRowIJ(Ao,one,zero,zero,n,ia,iia,ja,jja,done,ierr))
      PetscCallA(MatSeqAIJGetArray(Ao,aa,aaa,ierr))
      do 30, i=1,n
        write(7+rank,*) 'row ',i+rstart,' number nonzeros ',ia(iia+i+1)-ia(iia+i)
        do 40, j=ia(iia+i),ia(iia+i+1)-1
          write(7+rank,*)'  ',j,icol(iicol+ja(jja+j))+1,aa(aaa+j)
 40     continue
 30   continue
      PetscCallA(MatRestoreRowIJ(Ao,one,zero,zero,n,ia,iia,ja,jja,done,ierr))
      PetscCallA(MatSeqAIJRestoreArray(Ao,aa,aaa,ierr))

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
