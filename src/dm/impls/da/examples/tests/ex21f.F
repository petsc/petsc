!
!
      program main
#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscda.h"

!         For testing purposes this example also creates a 
!   DA context. Actually codes using SDA routines will probably 
!   not also work with DA contexts.


      integer        MM,ierr,dof,stencil_width,flg,i,start,end,PP
      integer        flg2,flg3,NN,m,n,p
      PetscOffset    in_idx,out_idx
      DAPeriodicType periodic
      DAStencilType  stencil_type
      DA             da
      integer        sda
      Vec            local,global
      Vec            local_copy
      PetscScalar    value,mone,in(1),out(1)
      PetscScalar    norm,work
 
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      m             = PETSC_DECIDE
      n             = PETSC_DECIDE
      p             = PETSC_DECIDE
      MM            = 8
      NN            = 6 
      PP            = 5
      dof           = 1
      stencil_width = 1
      periodic      = DA_NONPERIODIC
      stencil_type  = DA_STENCIL_STAR


      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-M',MM,flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-N',NN,flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-dof',dof,flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-stencil_width',         &
     &     stencil_width,flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-periodic',periodic,     &
     &     flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-stencil_type',          &
     &     stencil_type,flg,ierr)

      call PetscOptionsHasName(PETSC_NULL_CHARACTER,'-2d',flg2,ierr)
      call PetscOptionsHasName(PETSC_NULL_CHARACTER,'-3d',flg3,ierr)
      if (flg2 .ne. 0) then
         call DACreate2d(PETSC_COMM_WORLD,periodic,stencil_type,        &
     &        MM,NN,m,n,dof,stencil_width,PETSC_NULL_INTEGER,           &
     &        PETSC_NULL_INTEGER,da,ierr)
         call SDACreate2d(PETSC_COMM_WORLD,periodic,stencil_type,       &
     &        MM,NN,m,n,dof,stencil_width,PETSC_NULL_INTEGER,           &
     &        PETSC_NULL_INTEGER,sda,ierr)
      else if (flg3 .ne. 0) then
         call DACreate3d(PETSC_COMM_WORLD,periodic,stencil_type,        &
     &        MM,NN,PP,m,n,p,dof,stencil_width,PETSC_NULL_INTEGER,      &
     &        PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,da,ierr)
         call SDACreate3d(PETSC_COMM_WORLD,periodic,stencil_type,       &
     &        MM,NN,PP,m,n,p,dof,stencil_width,PETSC_NULL_INTEGER,      &
     &        PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,sda,ierr)
      else
         call DACreate1d(PETSC_COMM_WORLD,periodic,MM,dof,              &
     &        stencil_width,PETSC_NULL,da,ierr)
         call SDACreate1d(PETSC_COMM_WORLD,periodic,MM,dof,             &
     &        stencil_width,PETSC_NULL_INTEGER,sda,ierr)
      endif

      call DACreateGlobalVector(da,global,ierr)
      call DACreateLocalVector(da,local,ierr)
      call VecDuplicate(local,local_copy,ierr)

  
!   zero out vectors so that ghostpoints are zero 
      value = 0.0
      call VecSet(local,value,ierr)
      call VecSet(local_copy,value,ierr)

      call VecGetOwnershipRange(global,start,end,ierr)
      do 10, i=start,end-1
        value = i + 1
        call VecSetValues(global,1,i,value,INSERT_VALUES,ierr)
 10   continue
      call VecAssemblyBegin(global,ierr)
      call VecAssemblyEnd(global,ierr)

      call DAGlobalToLocalBegin(da,global,INSERT_VALUES,local,          &
     &                          ierr)
      call DAGlobalToLocalEnd(da,global,INSERT_VALUES,local,ierr)


      call VecGetArray(local,out,out_idx,ierr)
      call VecGetArray(local_copy,in,in_idx,ierr)
      call SDALocalToLocalBegin(sda,out(out_idx+1),INSERT_VALUES,       &
     &                          in(in_idx+1),ierr)
      call SDALocalToLocalEnd(sda,out(out_idx+1),INSERT_VALUES,         &
     &                        in(in_idx+1),ierr)

      mone = -1.0
      call VecAXPY(local_copy,mone,local,ierr)
      call VecNorm(local_copy,NORM_MAX,work,ierr)
      call MPI_Allreduce(work, norm,1,MPIU_REAL_PRECISION,MPI_MAX,      &
     &                   PETSC_COMM_WORLD,ierr)
      print*,'Norm of difference ',norm,' should be zero'
   
      call DADestroy(da,ierr)
      call SDADestroy(sda,ierr)
      call VecDestroy(local_copy,ierr)

      call VecDestroy(local,ierr)
      call VecDestroy(global,ierr)
      call PetscFinalize(ierr)
      end
