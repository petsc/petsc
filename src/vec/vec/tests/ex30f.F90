!
!
!  Tests parallel to parallel scatter where a to from index are
!  duplicated
      program main
#include <petsc/finclude/petscvec.h>
      use petscvec
      implicit none

      PetscErrorCode ierr
      PetscInt  nlocal, n, row
      PetscInt  nlocal2,n2,eight
      PetscMPIInt rank, size
      PetscInt from(10), to(10)

      PetscScalar num
      Vec v1, v2, v3
      VecScatter scat1, scat2
      IS fromis, tois
      n=8
      nlocal=2
      PetscCallA(PetscInitialize(ierr))
      PetscCallMPIA(MPI_COMM_RANK(PETSC_COMM_WORLD,rank,ierr))
      PetscCallMPIA(MPI_COMM_SIZE(PETSC_COMM_WORLD,size,ierr))
      if (size.ne.4) then
         print *, 'Four processor test'
         stop
      end if

      nlocal2 = 2*nlocal
      n2      = 2*n
      PetscCallA(VecCreateMPI(PETSC_COMM_WORLD,nlocal2,n2,v1,ierr))
      PetscCallA(VecCreateMPI(PETSC_COMM_WORLD,nlocal,n,v2,ierr))
      PetscCallA(VecCreateSeq(PETSC_COMM_SELF,n,v3,ierr))

      num=2.0
      row = 1
      PetscCallA(VecSetValue(v1,row,num,INSERT_VALUES,ierr))
      row = 5
      PetscCallA(VecSetValue(v1,row,num,INSERT_VALUES,ierr))
      row = 9
      PetscCallA(VecSetValue(v1,row,num,INSERT_VALUES,ierr))
      row = 13
      PetscCallA(VecSetValue(v1,row,num,INSERT_VALUES,ierr))
      num=1.0
      row = 15
      PetscCallA(VecSetValue(v1,row,num,INSERT_VALUES,ierr))
      row = 3
      PetscCallA(VecSetValue(v1,row,num,INSERT_VALUES,ierr))
      row = 7
      PetscCallA(VecSetValue(v1,row,num,INSERT_VALUES,ierr))
      row = 11
      PetscCallA(VecSetValue(v1,row,num,INSERT_VALUES,ierr))

      PetscCallA(VecAssemblyBegin(v1,ierr))
      PetscCallA(VecAssemblyEnd(v1,ierr))

      num=0.0
      PetscCallA(VecScale(v2,num,ierr))
      PetscCallA(VecScale(v3,num,ierr))

      from(1)=1
      from(2)=5
      from(3)=9
      from(4)=13
      from(5)=3
      from(6)=7
      from(7)=11
      from(8)=15
      to(1)=0
      to(2)=0
      to(3)=0
      to(4)=0
      to(5)=7
      to(6)=7
      to(7)=7
      to(8)=7

      eight = 8
      PetscCallA(ISCreateGeneral(PETSC_COMM_SELF,eight,from,PETSC_COPY_VALUES,fromis,ierr))
      PetscCallA(ISCreateGeneral(PETSC_COMM_SELF,eight,to,PETSC_COPY_VALUES,tois,ierr))
      PetscCallA(VecScatterCreate(v1,fromis,v2,tois,scat1,ierr))
      PetscCallA(VecScatterCreate(v1,fromis,v3,tois,scat2,ierr))
      PetscCallA(ISDestroy(fromis,ierr))
      PetscCallA(ISDestroy(tois,ierr))

      PetscCallA(VecScatterBegin(scat1,v1,v2,ADD_VALUES,SCATTER_FORWARD,ierr))
      PetscCallA(VecScatterEnd(scat1,v1,v2,ADD_VALUES,SCATTER_FORWARD,ierr))

      PetscCallA(VecScatterBegin(scat2,v1,v3,ADD_VALUES,SCATTER_FORWARD,ierr))
      PetscCallA(VecScatterEnd(scat2,v1,v3,ADD_VALUES,SCATTER_FORWARD,ierr))

      PetscCallA(PetscObjectSetName(v1, 'V1',ierr))
      PetscCallA(VecView(v1,PETSC_VIEWER_STDOUT_WORLD,ierr))

      PetscCallA(PetscObjectSetName(v2, 'V2',ierr))
      PetscCallA(VecView(v2,PETSC_VIEWER_STDOUT_WORLD,ierr))

      if (rank.eq.0) then
         PetscCallA(PetscObjectSetName(v3, 'V3',ierr))
         PetscCallA(VecView(v3,PETSC_VIEWER_STDOUT_SELF,ierr))
      end if

      PetscCallA(VecScatterDestroy(scat1,ierr))
      PetscCallA(VecScatterDestroy(scat2,ierr))
      PetscCallA(VecDestroy(v1,ierr))
      PetscCallA(VecDestroy(v2,ierr))
      PetscCallA(VecDestroy(v3,ierr))

      PetscCallA(PetscFinalize(ierr))

      end

!/*TEST
!
!     test:
!       nsize: 4
!
!TEST*/

