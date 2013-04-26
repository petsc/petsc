!
!       Formatted test for TS routines.
!
!          Solves U_t = U_xx 
!     F(t,u) = (u_i+1 - 2u_i + u_i-1)/h^2
!       using several different schemes. 
!
!23456789012345678901234567890123456789012345678901234567890123456789012

! This example need rewritten using newly developed TS interface!

      program main
      implicit none
#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscmat.h>
#include <finclude/petscdmda.h>
#include <finclude/petscpc.h>
#include <finclude/petscksp.h>
#include <finclude/petscsnes.h>
#include <finclude/petscts.h>
#include <finclude/petscdraw.h>
#include <finclude/petscviewer.h>

#include "ex1f.h"

      integer   linear_no_matrix,linear_no_time,linear
      integer   nonlinear_no_jacobian,nonlinear
      parameter (linear_no_matrix = 0,linear_no_time = 1,linear = 2)
      parameter (nonlinear_no_jacobian = 3,nonlinear = 4)

      PetscErrorCode  ierr
      PetscInt time_steps,steps
      PetscMPIInt size
      integer problem
      Vec              local,global
      double precision dt,ftime,zero,tmax
      TS               ts
      Mat              A
      MatStructure     A_structure
      TSProblemType    tsproblem
      PetscDraw        draw
      PetscViewer      viewer
      character*(60)   type,tsinfo
      character*(5)    etype
      PetscInt         i1,i60
      PetscBool  flg
      PetscSizeT five

      external Monitor,RHSFunctionHeat,RHSMatrixFree,Initial
      external RHSMatrixHeat,RHSJacobianHeat

      i1         = 1
      i60        = 60
      zero       = 0.0
      time_steps = 100
      problem    = linear_no_matrix
      A          = 0
      tsproblem  = TS_LINEAR
 
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)

      M = 60
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-M',M,flg,ierr) 
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-time',time_steps,   &
     &     flg,ierr)

      call PetscOptionsHasName(PETSC_NULL_CHARACTER,'-nox',flg,ierr) 
      if (flg) then
        nox = 1 
      else
        nox = 0
      endif
      nrm_2   = 0.0 
      nrm_max = 0.0

!   Set up the ghost point communication pattern                       

      call DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,M,i1,i1,            &
     &     PETSC_NULL_INTEGER,da,ierr)
      call DMCreateGlobalVector(da,global,ierr) 
      call VecGetLocalSize(global,m,ierr) 
      call DMCreateLocalVector(da,local,ierr) 

!   Set up display to show wave graph 

      call PetscViewerDrawOpen(PETSC_COMM_WORLD,PETSC_NULL_CHARACTER,        &
     &     PETSC_NULL_CHARACTER,80,380,400,160,viewer1,ierr)
      call PetscViewerDrawGetDraw(viewer1,0,draw,ierr) 
      call PetscDrawSetDoubleBuffer(draw,ierr) 
      call PetscViewerDrawOpen(PETSC_COMM_WORLD,PETSC_NULL_CHARACTER,        &
     &     PETSC_NULL_CHARACTER,80,0,400,160,viewer2,ierr) 
      call PetscViewerDrawGetDraw(viewer2,0,draw,ierr) 
      call PetscDrawSetDoubleBuffer(draw,ierr) 

!   make work array for evaluating right hand side function 

      call VecDuplicate(local,localwork,ierr) 

!   make work array for storing exact solution 

      call VecDuplicate(global,csolution,ierr) 

      h = 1.0/(M-1.0)

!   set initial conditions
 
      call Initial(global,PETSC_NULL_OBJECT,ierr) 
 
!
!     This example is written to allow one to easily test parts 
!    of TS, we do not expect users to generally need to use more
!    then a single TSProblemType
!
      call PetscOptionsHasName(PETSC_NULL_CHARACTER,'-linear_no_matrix',&
     &                    flg,ierr) 
      if (flg) then
        tsproblem = TS_LINEAR
        problem   = linear_no_matrix
      endif
      call PetscOptionsHasName(PETSC_NULL_CHARACTER,                    &
     &     '-linear_constant_matrix',flg,ierr) 
      if (flg) then
        tsproblem = TS_LINEAR
        problem   = linear_no_time
      endif
      call PetscOptionsHasName(PETSC_NULL_CHARACTER,                    &
     &     '-nonlinear_no_jacobian',flg,ierr) 
      if (flg) then
        tsproblem = TS_NONLINEAR
        problem   = nonlinear_no_jacobian
      endif
      call PetscOptionsHasName(PETSC_NULL_CHARACTER,                    &
     &     '-nonlinear_jacobian',flg,ierr) 
      if (flg) then
        tsproblem = TS_NONLINEAR
        problem   = nonlinear
      endif
    
!   make timestep context 

      call TSCreate(PETSC_COMM_WORLD,ts,ierr) 
      call TSSetProblemType(ts,tsproblem,ierr) 
      call TSMonitorSet(ts,Monitor,PETSC_NULL_OBJECT,                   &
     &                  PETSC_NULL_FUNCTION, ierr) 

      dt = h*h/2.01

      if (problem .eq. linear_no_matrix) then
!    
!         The user provides the RHS as a Shell matrix.
!    
         call MatCreateShell(PETSC_COMM_WORLD,m,M,M,M,                  &
     &        PETSC_NULL_OBJECT,A,ierr)
         call MatShellSetOperation(A,MATOP_MULT,RHSMatrixFree,ierr)
         call TSSetRHSJacobian(ts,A,A,TSComputeRHSJacobianConstant,     &
     &        PETSC_NULL_OBJECT,ierr)
         call TSSetRHSFunction(ts,PETSC_NULL_OBJECT,                    &
     &        TSComputeRHSFunctionLinear,PETSC_NULL_OBJECT,ierr)
      else if (problem .eq. linear_no_time) then
!    
!         The user provides the RHS as a matrix
!    
         call MatCreate(PETSC_COMM_WORLD,A,ierr) 
         call MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,M,ierr) 
         call MatSetFromOptions(A,ierr)
         call RHSMatrixHeat(ts,zero,A,A,A_structure,PETSC_NULL_OBJECT,  &
     &        ierr)  
         call TSSetRHSJacobian(ts,A,A,TSComputeRHSJacobianConstant,     &
     &        PETSC_NULL_OBJECT,ierr)
         call TSSetRHSFunction(ts,PETSC_NULL_OBJECT,                    &
     &        TSComputeRHSFunctionLinear,PETSC_NULL_OBJECT,ierr)
      else if (problem .eq. nonlinear_no_jacobian) then
!    
!         The user provides the RHS and a Shell Jacobian
!    
         call TSSetRHSFunction(ts,PETSC_NULL_OBJECT,RHSFunctionHeat,    &
     &        PETSC_NULL_OBJECT,ierr)
         call MatCreateShell(PETSC_COMM_WORLD,m,M,M,M,                  &
     &        PETSC_NULL_OBJECT,A,ierr)
         call MatShellSetOperation(A,MATOP_MULT,RHSMatrixFree,ierr)
         call TSSetRHSJacobian(ts,A,A,TSComputeRHSJacobianConstant,     &
     &        PETSC_NULL_OBJECT,ierr) 
      else if (problem .eq. nonlinear) then
!    
!         The user provides the RHS and Jacobian
!    
         call TSSetRHSFunction(ts,PETSC_NULL_OBJECT,RHSFunctionHeat,    &
     &        PETSC_NULL_OBJECT,ierr)
         call MatCreate(PETSC_COMM_WORLD,A,ierr) 
         call MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,M,ierr) 
         call MatSetFromOptions(A,ierr)
         call RHSMatrixHeat(ts,zero,A,A,A_structure,PETSC_NULL_OBJECT,   &
     &        ierr)  
         call TSSetRHSJacobian(ts,A,A,RHSJacobianHeat,                   &
     &        PETSC_NULL_OBJECT,ierr)
      endif

      call TSSetFromOptions(ts,ierr)

      call TSSetInitialTimeStep(ts,zero,dt,ierr) 
      tmax = 100.0
      call TSSetDuration(ts,time_steps,tmax,ierr) 
      call TSSetSolution(ts,global,ierr) 

      call TSSetUp(ts,ierr) 
      call TSStep(ts,steps,ftime,ierr) 
      call PetscViewerStringOpen(PETSC_COMM_WORLD,tsinfo,i60,viewer,         &
     &                           ierr) 
      call TSView(ts,viewer,ierr) 
      call TSGetType(ts,type,ierr)

      call PetscOptionsHasName(PETSC_NULL_CHARACTER,'-test',flg,ierr) 
      if (flg) then
!   
!         copy type to string of length 5 to ensure equality test        
!         is done correctly
!     
        five = 5
        call PetscStrncpy(etype,type,five,ierr)
        if (etype .eq. TSEULER) then
          if (abs(nrm_2/steps - 0.00257244) .gt. 1.e-4) then
            print*,'Error in Euler method: 2-norm ',nrm_2/steps,        &
     &            ' expecting: ',0.00257244
          endif
        else
          if (abs(nrm_2/steps - 0.00506174) .gt. 1.e-4) then
            print*,'Error in ',tsinfo,': 2-norm ',nrm_2/steps,          &
     &             ' expecting: ',0.00506174
          endif
        endif
      else 
        print*,size,' Procs Avg. error 2 norm ',nrm_2/steps,            &
     &              nrm_max/steps,tsinfo
      endif

      call PetscViewerDestroy(viewer,ierr) 
      call TSDestroy(ts,ierr) 
      call PetscViewerDestroy(viewer1,ierr) 
      call PetscViewerDestroy(viewer2,ierr) 
      call VecDestroy(localwork,ierr) 
      call VecDestroy(csolution,ierr)
      call VecDestroy(local,ierr) 
      call VecDestroy(global,ierr) 
      call DMDestroy(da,ierr) 
      if (A .ne. 0) then
        call MatDestroy(A,ierr) 
      endif

      call PetscFinalize(ierr)
      end

!  -------------------------------------------------------------------
 
      subroutine Initial(global,ctx,ierr)
      implicit none
#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscmat.h>
#include <finclude/petscdmda.h>
#include <finclude/petscpc.h>
#include <finclude/petscksp.h>
#include <finclude/petscsnes.h>
#include <finclude/petscts.h>
#include <finclude/petscviewer.h>

#include "ex1f.h"

      Vec         global
      PetscObject    ctx

      PetscScalar localptr(1)
      PetscInt     i,mybase,myend
      PetscErrorCode ierr
      PetscOffset idx

!   determine starting point of each processor 

      call VecGetOwnershipRange(global,mybase,myend,ierr) 

!   Initialize the array 

      call VecGetArray(global,localptr,idx,ierr) 
      do 10, i=mybase,myend-1
        localptr(i-mybase+1+idx) = sin(PETSC_PI*i*6.*h) +               &
     &                             3.*sin(PETSC_PI*i*2.*h)
 10   continue
      call VecRestoreArray(global,localptr,idx,ierr) 
      return
      end

! ------------------------------------------------------------------------------
! 
!       Exact solution 
!  
      subroutine Solution(t,sol,ctx)
      implicit none
#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscmat.h>
#include <finclude/petscdmda.h>
#include <finclude/petscpc.h>
#include <finclude/petscksp.h>
#include <finclude/petscsnes.h>
#include <finclude/petscts.h>
#include <finclude/petscviewer.h>

#include "ex1f.h"

      double precision  t
      Vec               sol
      PetscObject       ctx

      PetscScalar localptr(1),ex1
      PetscScalar ex2,sc1,sc2
      PetscInt    i,mybase,myend
      PetscErrorCode ierr
      PetscOffset       idx

!   determine starting point of each processor 

      call VecGetOwnershipRange(csolution,mybase,myend,ierr) 

      ex1 = exp(-36.*PETSC_PI*PETSC_PI*t) 
      ex2 = exp(-4.*PETSC_PI*PETSC_PI*t)
      sc1 = PETSC_PI*6.*h
      sc2 = PETSC_PI*2.*h
      call VecGetArray(csolution,localptr,idx,ierr) 
      do 10, i=mybase,myend-1
        localptr(i-mybase+1+idx) = sin(i*sc1)*ex1 + 3.*sin(i*sc2)*ex2
 10   continue
      call VecRestoreArray(csolution,localptr,idx,ierr) 
      return   
      end 


! -----------------------------------------------------------------------------------

      subroutine Monitor(ts,step,time,global,ctx,ierr)
      implicit none
#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscmat.h>
#include <finclude/petscdmda.h>
#include <finclude/petscpc.h>
#include <finclude/petscksp.h>
#include <finclude/petscsnes.h>
#include <finclude/petscts.h>
#include <finclude/petscdraw.h>
#include <finclude/petscviewer.h>

#include "ex1f.h"
      TS                ts
      PetscInt           step
      PetscObject     ctx
      PetscErrorCode ierr
      double precision  time,lnrm_2,lnrm_max
      Vec               global
      PetscScalar       mone

      call VecView(global,viewer1,ierr) 

      call Solution(time,csolution,ctx) 
      mone = -1.0
      call VecAXPY(csolution,mone,global,ierr) 
      call VecNorm(csolution,NORM_2,lnrm_2,ierr) 
      lnrm_2 = sqrt(h)*lnrm_2
      call VecNorm(csolution,NORM_MAX,lnrm_max,ierr) 

      if (nox .eq. 0) then
        print*,'timestep ',step,' time ',time,' norm of error ',        &
     &         lnrm_2,lnrm_max
      endif

      nrm_2   = nrm_2 + lnrm_2
      nrm_max = nrm_max +  lnrm_max
      call VecView(csolution,viewer2,ierr) 

      return   
      end

!  -----------------------------------------------------------------------

      subroutine RHSMatrixFree(mat,x,y,ierr)
      implicit none
#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscmat.h>
#include <finclude/petscdmda.h>
#include <finclude/petscpc.h>
#include <finclude/petscksp.h>
#include <finclude/petscsnes.h>
#include <finclude/petscts.h>
#include <finclude/petscviewer.h>
      Mat              mat
      Vec              x,y
      PetscErrorCode          ierr
      double precision zero
      TS               ts0

      zero = 0.0

      ts0 = PETSC_NULL_OBJECT

      call RHSFunctionHeat(ts0,zero,x,y,PETSC_NULL_OBJECT,ierr) 
      return
      end

! -------------------------------------------------------------------------

      subroutine RHSFunctionHeat(ts,t,globalin,globalout,ctx,ierr)
      implicit none
#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscmat.h>
#include <finclude/petscdmda.h>
#include <finclude/petscpc.h>
#include <finclude/petscksp.h>
#include <finclude/petscsnes.h>
#include <finclude/petscts.h>
#include <finclude/petscviewer.h>

#include "ex1f.h"
      TS               ts
      double precision t
      Vec globalin,globalout
      PetscObject ctx
      Vec local
      PetscErrorCode  ierr
      PetscInt i,localsize
      PetscOffset ldx,cdx 
      PetscScalar copyptr(1),localptr(1)
      PetscScalar sc

!  Extract local array  

      call DMCreateLocalVector(da,local,ierr) 
      call DMGlobalToLocalBegin(da,globalin,INSERT_VALUES,local,ierr) 
      call DMGlobalToLocalEnd(da,globalin,INSERT_VALUES,local,ierr) 
      call VecGetArray(local,localptr,ldx,ierr) 

!  Extract work vector 

      call VecGetArray(localwork,copyptr,cdx,ierr) 

!   Update Locally - Make array of new values 
!   Note: For the first and last entry I copy the value 
!   if this is an interior node it is irrelevant 

      sc = 1.0/(h*h)
      call VecGetLocalSize(local,localsize,ierr) 
      copyptr(1+cdx) = localptr(1+ldx)
      do 10, i=1,localsize-2
        copyptr(i+1+cdx) = sc * (localptr(i+2+ldx) + localptr(i+ldx) -  &
     &                     2.0*localptr(i+1+ldx))
 10   continue
      copyptr(localsize-1+1+cdx) = localptr(localsize-1+1+ldx)
      call VecRestoreArray(localwork,copyptr,cdx,ierr) 
      call VecRestoreArray(local,localptr,ldx,ierr) 
      call VecDestroy(local,ierr)

!   Local to Global 

      call DMLocalToGlobalBegin(da,localwork,INSERT_VALUES,globalout,    &
     &             ierr) 
      call DMLocalToGlobalEnd(da,localwork,INSERT_VALUES,globalout,      &
     &            ierr) 
      return 
      end

!  ---------------------------------------------------------------------

      subroutine RHSMatrixHeat(ts,t,AA,BB,str,ctx,ierr)
      implicit none
#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscmat.h>
#include <finclude/petscdmda.h>
#include <finclude/petscpc.h>
#include <finclude/petscksp.h>
#include <finclude/petscsnes.h>
#include <finclude/petscts.h>
#include <finclude/petscviewer.h>

#include "ex1f.h"
      Mat              AA,BB
      double precision t
      TS               ts
      MatStructure     str
      PetscObject          ctx
      PetscErrorCode ierr

      Mat              A
      PetscInt    i,mstart(1),mend(1),idx(3)
      PetscMPIInt rank,size
      PetscScalar v(3),stwo,sone
      PetscInt    i1,i3

      i1 = 1
      i3 = 3
      A    = AA
      stwo = -2./(h*h)
      sone = -.5*stwo
      str  = SAME_NONZERO_PATTERN

      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)

      call MatGetOwnershipRange(A,mstart,mend,ierr) 
      if (mstart(1) .eq. 0) then
        v(1) = 1.0
        call MatSetValues(A,i1,mstart(1),i1,mstart(1),v,INSERT_VALUES,  &
     &       ierr) 
        mstart(1) = mstart(1) + 1
      endif
      if (mend(1) .eq. M) then
        mend(1) = mend(1) - 1
        v(1) = 1.0
        call MatSetValues(A,i1,mend,i1,mend,v,INSERT_VALUES,ierr) 
      endif

!  
!     Construct matrice one row at a time
!  
      v(1) = sone
      v(2) = stwo
      v(3) = sone  
      do 10, i=mstart(1),mend(1)-1
        idx(1) = i-1
        idx(2) = i 
        idx(3) = i+1
        call MatSetValues(A,i1,i,i3,idx,v,INSERT_VALUES,ierr) 
 10   continue

      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr) 
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr) 
      return
      end

! --------------------------------------------------------------------------------------

      subroutine RHSJacobianHeat(ts,t,x,AA,BB,str,ctx,ierr)
      implicit none
#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscmat.h>
#include <finclude/petscdmda.h>
#include <finclude/petscpc.h>
#include <finclude/petscksp.h>
#include <finclude/petscsnes.h>
#include <finclude/petscts.h>
#include <finclude/petscviewer.h>
      TS               ts
      double precision t
      Vec              x
      Mat              AA,BB
      MatStructure     str
      PetscObject      ctx
      PetscErrorCode ierr

      call RHSMatrixHeat(ts,t,AA,BB,str,ctx,ierr)
      return
      end
