
!      "$Id: ex79f.F,v 1.3 2001/08/07 03:03:07 balay Exp $";
!
!   This program demonstrates use of MatGetRowIJ() from Fortran
!
      program main
      implicit none
#include "include/finclude/petsc.h"
#include "include/finclude/petscmat.h"
#include "include/finclude/petscis.h"
#include "include/finclude/petscviewer.h"

      Mat         A,Ad,Ao
      integer     ierr,rank
      PetscViewer v    
      integer     i,j,ia(1),ja(1),n,icol(1),rstart,rend
      PetscTruth  done
      PetscOffset iia,jja,aaa,iicol
      PetscScalar aa(1)

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)

      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'../matbinary.ex',         &
     &                          PETSC_FILE_RDONLY,v,ierr)
      call MatLoad(v,MATMPIAIJ,A,ierr)
      call MatView(A,PETSC_VIEWER_STDOUT_WORLD,ierr)

      call MatMPIAIJGetSeqAIJ(A,Ad,Ao,icol,iicol,ierr)
      call MatGetOwnershipRange(A,rstart,rend,ierr)
!
!   Print diagonal portion of matrix
!
      call PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1,ierr)
      call MatGetRowIJ(Ad,1,0,n,ia,iia,ja,jja,done,ierr)
      call MatGetArray(Ad,aa,aaa,ierr)
      do 10, i=1,n
        write(7+rank,*) 'row ',i+rstart,' number nonzeros ',                &
     &                   ia(iia+i+1)-ia(iia+i)
        do 20, j=ia(iia+i),ia(iia+i+1)-1
          write(7+rank,*)'  ',j,ja(jja+j)+rstart,aa(aaa+j)
 20     continue
 10   continue
      call MatRestoreRowIJ(Ad,1,0,n,ia,iia,ja,jja,done,ierr)

!
!   Print off-diagonal portion of matrix
!
      call MatGetRowIJ(Ao,1,0,n,ia,iia,ja,jja,done,ierr)
      call MatGetArray(Ao,aa,aaa,ierr)
      do 30, i=1,n
        write(7+rank,*) 'row ',i+rstart,' number nonzeros ',               &
     &                  ia(iia+i+1)-ia(iia+i)
        do 40, j=ia(iia+i),ia(iia+i+1)-1
          write(7+rank,*)'  ',j,icol(iicol+ja(jja+j))+1,aa(aaa+j)
 40     continue
 30   continue
      call MatRestoreRowIJ(Ao,1,0,n,ia,iia,ja,jja,done,ierr)
      call PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1,ierr)

      call MatView(A,PETSC_VIEWER_STDOUT_WORLD,ierr)
      call MatDestroy(A,ierr)
      call PetscViewerDestroy(v,ierr)
      call PetscFinalize(ierr)
      end




