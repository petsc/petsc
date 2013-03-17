!
!
!  Tests parallel to parallel scatter where a to from index are
!  duplicated
      program main
      implicit none

#include <finclude/petscsys.h>
#include <finclude/petscis.h>
#include <finclude/petscvec.h>

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
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_COMM_RANK(PETSC_COMM_WORLD,rank,ierr)
      call MPI_COMM_SIZE(PETSC_COMM_WORLD,size,ierr)
      if (size.ne.4) then
         print *, 'Four processor test'
         stop
      end if

      nlocal2 = 2*nlocal
      n2      = 2*n
      call VecCreateMPI(PETSC_COMM_WORLD,nlocal2,n2,v1,ierr)
      call VecCreateMPI(PETSC_COMM_WORLD,nlocal,n,v2,ierr)
      call VecCreateSeq(PETSC_COMM_SELF,n,v3,ierr)

      num=2.0
      row = 1
      call VecSetValue(v1,row,num,INSERT_VALUES,ierr)
      row = 5
      call VecSetValue(v1,row,num,INSERT_VALUES,ierr)
      row = 9
      call VecSetValue(v1,row,num,INSERT_VALUES,ierr)
      row = 13
      call VecSetValue(v1,row,num,INSERT_VALUES,ierr)
      num=1.0
      row = 15
      call VecSetValue(v1,row,num,INSERT_VALUES,ierr)
      row = 3
      call VecSetValue(v1,row,num,INSERT_VALUES,ierr)
      row = 7
      call VecSetValue(v1,row,num,INSERT_VALUES,ierr)
      row = 11
      call VecSetValue(v1,row,num,INSERT_VALUES,ierr)

      call VecAssemblyBegin(v1,ierr)
      call VecAssemblyEnd(v1,ierr)

      num=0.0
      call VecScale(v2,num,ierr)
      call VecScale(v3,num,ierr)

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
      call ISCreateGeneral(PETSC_COMM_SELF,eight,from,PETSC_COPY_VALUES,      &
     &                     fromis,ierr)
      call ISCreateGeneral(PETSC_COMM_SELF,eight,to,PETSC_COPY_VALUES,        &
     &                     tois,ierr)
      call VecScatterCreate(v1,fromis,v2,tois,scat1,ierr)
      call VecScatterCreate(v1,fromis,v3,tois,scat2,ierr)
      call ISDestroy(fromis,ierr)
      call ISDestroy(tois,ierr)

      call VecScatterBegin(scat1,v1,v2,ADD_VALUES,SCATTER_FORWARD,ierr)
      call VecScatterEnd(scat1,v1,v2,ADD_VALUES,SCATTER_FORWARD,ierr)

      call VecScatterBegin(scat2,v1,v3,ADD_VALUES,SCATTER_FORWARD,ierr)
      call VecScatterEnd(scat2,v1,v3,ADD_VALUES,SCATTER_FORWARD,ierr)

      call PetscObjectSetName(v1, "V1",ierr)
      call VecView(v1,PETSC_VIEWER_STDOUT_WORLD,ierr)

      call PetscObjectSetName(v2, "V2",ierr)
      call VecView(v2,PETSC_VIEWER_STDOUT_WORLD,ierr)

      if (rank.eq.0) then
         call PetscObjectSetName(v3, "V3",ierr)
         call VecView(v3,PETSC_VIEWER_STDOUT_SELF,ierr)
      end if

      call VecScatterDestroy(scat1,ierr)
      call VecScatterDestroy(scat2,ierr)
      call VecDestroy(v1,ierr)
      call VecDestroy(v2,ierr)
      call VecDestroy(v3,ierr)

      call PetscFinalize(ierr)

      end



