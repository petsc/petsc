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

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)

      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,                          &
     & '${PETSC_DIR}/share/petsc/datafiles/matrices/' //                       &
     & 'ns-real-int32-float64',                                               &
     &                          FILE_MODE_READ,v,ierr)
      call MatCreate(PETSC_COMM_WORLD,A,ierr)
      call MatSetType(A, MATMPIAIJ,ierr)
      call MatLoad(A,v,ierr)
      CHKERRA(ierr)
      call MatView(A,PETSC_VIEWER_STDOUT_WORLD,ierr)

      call MatMPIAIJGetSeqAIJ(A,Ad,Ao,icol,iicol,ierr)
      call MatGetOwnershipRange(A,rstart,rend,ierr)
!
!   Print diagonal portion of matrix
!
      call PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1,ierr)
      zero = 0
      one  = 1
      call MatGetRowIJ(Ad,one,zero,zero,n,ia,iia,ja,jja,done,ierr)
      call MatSeqAIJGetArray(Ad,aa,aaa,ierr)
      do 10, i=1,n
        write(7+rank,*) 'row ',i+rstart,' number nonzeros ',                &
     &                   ia(iia+i+1)-ia(iia+i)
        do 20, j=ia(iia+i),ia(iia+i+1)-1
          write(7+rank,*)'  ',j,ja(jja+j)+rstart,aa(aaa+j)
 20     continue
 10   continue
      call MatRestoreRowIJ(Ad,one,zero,zero,n,ia,iia,ja,jja,done,ierr)
      call MatSeqAIJRestoreArray(Ad,aa,aaa,ierr)
!
!   Print off-diagonal portion of matrix
!
      call MatGetRowIJ(Ao,one,zero,zero,n,ia,iia,ja,jja,done,ierr)
      call MatSeqAIJGetArray(Ao,aa,aaa,ierr)
      do 30, i=1,n
        write(7+rank,*) 'row ',i+rstart,' number nonzeros ',               &
     &                  ia(iia+i+1)-ia(iia+i)
        do 40, j=ia(iia+i),ia(iia+i+1)-1
          write(7+rank,*)'  ',j,icol(iicol+ja(jja+j))+1,aa(aaa+j)
 40     continue
 30   continue
      call MatRestoreRowIJ(Ao,one,zero,zero,n,ia,iia,ja,jja,done,ierr)
      call MatSeqAIJRestoreArray(Ao,aa,aaa,ierr)

      call PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1,ierr)

      call MatGetDiagonalBlock(A,Ad,ierr)
      call MatView(Ad,PETSC_VIEWER_STDOUT_WORLD,ierr)

      call MatView(A,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call MatDestroy(A,ierr)
      call PetscViewerDestroy(v,ierr)
      call PetscFinalize(ierr)
      end

!/*TEST
!
!     build:
!       requires: double !complex !define(PETSC_USE_64BIT_INDICES)
!
!     test:
!        args: -binary_read_double -options_left false
!
!TEST*/
