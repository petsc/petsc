
/* "$Id: ex2.F,v 1.1 1996/02/05 04:45:59 bsmith Exp bsmith $"; */


#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/da.h"

C         For testing purposes this example also creates a 
C   DA context. Actually codes using SDA routines will probably 
C   not also work with DA contexts.


      integer        MM,ierr,dof,stencil_width,flg,i,start,end,PP
      integer        flg2,flg3,NN,m,n,p,in_idx,out_idx
      DAPeriodicType periodic
      DAStencilType  stencil_type
      DA             da
      integer        sda
      Vec            local,global,local_copy
      Scalar         value,mone,in(1),out(1)
      Scalar         norm,work
 
      call PetscInitialize(ierr)

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


      call OptionsGetInt(PETSC_NULL_CHAR,'-M',MM,flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHAR,'-N',NN,flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHAR,'-dof',dof,flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHAR,'-stencil_width',
     *                   stencil_width,flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHAR,'-periodic',periodic,
     *                   flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHAR,'-stencil_type',
     *                   stencil_type,flg,ierr)

      call OptionsHasName(PETSC_NULL,'-2d',flg2,ierr)
      call OptionsHasName(PETSC_NULL,'-3d',flg3,ierr)
      if (flg2 .ne. 0) then
        call DACreate2d(MPI_COMM_WORLD,periodic,stencil_type,
     *                  MM,NN,m,n,dof,stencil_width,da,ierr)
        call SDACreate2d(MPI_COMM_WORLD,periodic,stencil_type,
     *                  MM,NN,m,n,dof,stencil_width,sda,ierr)
      else if (flg3 .ne. 0) then
        call DACreate3d(MPI_COMM_WORLD,periodic,stencil_type,
     *                  MM,NN,PP,m,n,p,dof,stencil_width,da,ierr)
        call SDACreate3d(MPI_COMM_WORLD,periodic,stencil_type,
     *                  MM,NN,PP,m,n,p,dof,stencil_width,sda,ierr)
      else
        call DACreate1d(MPI_COMM_WORLD,periodic,MM,dof,
     *                  stencil_width,da,ierr)
        call SDACreate1d(MPI_COMM_WORLD,periodic,MM,dof,
     *                  stencil_width,sda,ierr)
      endif

      call DAGetDistributedVector(da,global,ierr)
      call DAGetLocalVector(da,local,ierr)
      call VecDuplicate(local,local_copy,ierr)

  
C   zero out vectors so that ghostpoints are zero 
      value = 0.0
      call VecSet(value,local,ierr)
      call VecSet(value,local_copy,ierr)

      call VecGetOwnershipRange(global,start,end,ierr)
      do 10, i=start,end-1
        value = i + 1
        call VecSetValues(global,1,i,value,INSERT_VALUES,ierr)
 10   continue
      call VecAssemblyBegin(global,ierr)
      call VecAssemblyEnd(global,ierr)

      call DAGlobalToLocalBegin(da,global,INSERT_VALUES,local,
     *                          ierr)
      call DAGlobalToLocalEnd(da,global,INSERT_VALUES,local,ierr)


      call VecGetArray(local,out,out_idx,ierr)
      call VecGetArray(local_copy,in,in_idx,ierr)
      call SDALocalToLocalBegin(sda,out(out_idx+1),INSERT_VALUES,
     *                          in(in_idx+1),ierr)
      call SDALocalToLocalEnd(sda,out(out_idx+1),INSERT_VALUES,
     *                        in(in_idx+1),ierr)

      mone = -1.0
      call VecAXPY(mone,local,local_copy,ierr)
      call VecNorm(local_copy,NORM_MAX,work,ierr)
      call MPI_Allreduce( work, norm,1,MPI_DOUBLE_PRECISION,MPI_MAX,
     *                   MPI_COMM_WORLD,ierr )
      print*,'Norm of difference ',norm,' should be zero'
   
      call DADestroy(da,ierr)
      call SDADestroy(sda,ierr)
      call VecDestroy(local_copy,ierr)

      call VecDestroy(local,ierr)
      call VecDestroy(global,ierr)
      call PetscFinalize()
      stop 
      end
