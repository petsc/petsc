C
C "$Id: ex1f.F,v 1.1 1996/09/26 03:11:57 bsmith Exp bsmith $";
C
C       Formatted test for TS routines.
C
C          Solves U_t = U_xx 
C     F(t,u) = (u_i+1 - 2u_i + u_i-1)/h^2
C       using several different schemes. 
C
C23456789012345678901234567890123456789012345678901234567890123456789012

#include "FINCLUDE/petsc.h"
#include "FINCLUDE/vec.h"
#include "FINCLUDE/mat.h"
#include "FINCLUDE/da.h"
#include "FINCLUDE/sys.h"
#include "FINCLUDE/sles.h"
#include "FINCLUDE/snes.h"
#include "FINCLUDE/ts.h"
#include "FINCLUDE/draw.h"
#include "FINCLUDE/viewer.h"

      integer   linear_no_matrix,linear_no_time,linear,
     &          nonlinear_no_jacobian,nonlinear
      parameter (linear_no_matrix = 0,linear_no_time = 1,linear = 2,
     &           nonlinear_no_jacobian = 3,nonlinear = 4)

      integer         ierr,  time_steps, steps, flg, size, m,problem
      Vec             local, global
      Double          dt,ftime
      TS              ts
      TSType          type
      Mat             A
      MatStructure    A_structure
      TSProblemType   tsproblem
      Draw            draw
      Viewer          viewer
      character*(60)  tsinfo

C
C    Application context data, stored in common block
C
      Vec            localwork,csolution
      DA             da
      Viewer         viewer1,viewer2
      integer        M
      Double         h,nrm_2,nrm_max,nox
      common /tsctx/ localwork,csolution,da,viewer1,viewer2,M,h,nrm_2,
     &               nrm_max,nox

      external Monitor, RHSFunctionHeat, RHSMatrixFree, Initial
      external RHSMatrixHeat, RHSJacobianHeat

      time_steps = 100
      problem    = linear_no_matrix
      A          = 0
      tsproblem  = TS_LINEAR
 
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)

      M = 60
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-M',M,flg,ierr) 
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-time',time_steps,flg,
     &                   ierr)

      call OptionsHasName(PETSC_NULL_CHARACTER,'-nox',flg,ierr) 
      if (flg .eq. 1) then
        nox = 1 
      else
        nox = 0
      endif
      nrm_2   = 0.0 
      nrm_max = 0.0

C   Set up the ghost point communication pattern                       

      call DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,M,1,1,da,ierr)
      call DAGetDistributedVector(da,global,ierr) 
      call VecGetLocalSize(global,m,ierr) 
      call DAGetLocalVector(da,local,ierr) 

C   Set up display to show wave graph 

      call ViewerDrawOpenX(PETSC_COMM_WORLD,PETSC_NULL_CHARACTER,
     &              PETSC_NULL_CHARACTER,80,380,400,160,viewer1,ierr)
      call ViewerDrawGetDraw(viewer1,draw,ierr) 
      call DrawSetDoubleBuffer(draw,ierr) 
      call ViewerDrawOpenX(PETSC_COMM_WORLD,PETSC_NULL_CHARACTER,
     &              PETSC_NULL_CHARACTER,80,0,400,160,viewer2,ierr) 
      call ViewerDrawGetDraw(viewer2,draw,ierr) 
      call DrawSetDoubleBuffer(draw,ierr) 

C   make work array for evaluating right hand side function 

      call VecDuplicate(local,localwork,ierr) 

C   make work array for storing exact solution 

      call VecDuplicate(global,csolution,ierr) 

      h = 1.0/(M-1.0)

C   set initial conditions
 
      call Initial(global,PETSC_NULL,ierr) 
 
C
C     This example is written to allow one to easily test parts 
C    of TS, we do not expect users to generally need to use more
C    then a single TSProblemType
C
      call OptionsHasName(PETSC_NULL_CHARACTER,'-linear_no_matrix',
     &                    flg,ierr) 
      if (flg .eq. 1) then
        tsproblem = TS_LINEAR
        problem   = linear_no_matrix
      endif
      call OptionsHasName(PETSC_NULL_CHARACTER,
     &                    '-linear_constant_matrix',flg,ierr) 
      if (flg .eq. 1) then
        tsproblem = TS_LINEAR
        problem   = linear_no_time
      endif
      call OptionsHasName(PETSC_NULL_CHARACTER,'-linear_variable_matrix'
     &                    ,flg,ierr) 
      if (flg .eq. 1) then
        tsproblem = TS_LINEAR
        problem   = linear
      endif
      call OptionsHasName(PETSC_NULL_CHARACTER,'-nonlinear_no_jacobian',
     &                    flg,ierr) 
      if (flg .eq. 1) then
        tsproblem = TS_NONLINEAR
        problem   = nonlinear_no_jacobian
      endif
      call OptionsHasName(PETSC_NULL_CHARACTER,'-nonlinear_jacobian',
     &                    flg,ierr) 
      if (flg .eq. 1) then
        tsproblem = TS_NONLINEAR
        problem   = nonlinear
      endif
    
C   make time step context 

      call TSCreate(PETSC_COMM_WORLD,tsproblem,ts,ierr) 
      call TSSetMonitor(ts,Monitor,PETSC_NULL,ierr) 

      dt = h*h/2.01

      if (problem .eq. linear_no_matrix) then
C    
C         The user provides the RHS as a Shell matrix.
C    
        call MatCreateShell(PETSC_COMM_WORLD,m,M,M,M,PETSC_NULL,A,ierr)
        call MatShellSetOperation(A,MAT_MULT,RHSMatrixFree,ierr)
        call TSSetRHSMatrix(ts,A,A,PETSC_NULL,PETSC_NULL,ierr) 
      else if (problem .eq. linear_no_time) then
C    
C         The user provides the RHS as a matrix
C    
        call MatCreate(PETSC_COMM_WORLD,M,M,A,ierr) 
        call RHSMatrixHeat(ts,0.d0,A,A,A_structure,PETSC_NULL,ierr)  
        call TSSetRHSMatrix(ts,A,A,PETSC_NULL,PETSC_NULL,ierr) 
      else if (problem .eq. linear) then
C    
C         The user provides the RHS as a time dependent matrix
C    
        call MatCreate(PETSC_COMM_WORLD,M,M,A,ierr) 
        call RHSMatrixHeat(ts,0.d0,A,A,A_structure,PETSC_NULL,ierr)  
        call TSSetRHSMatrix(ts,A,A,RHSMatrixHeat,PETSC_NULL,ierr) 
      else if (problem .eq. nonlinear_no_jacobian) then
C    
C         The user provides the RHS and a Shell Jacobian
C    
        call TSSetRHSFunction(ts,RHSFunctionHeat,PETSC_NULL,ierr) 
        call MatCreateShell(PETSC_COMM_WORLD,m,M,M,M,PETSC_NULL,A,ierr)
        call MatShellSetOperation(A,MAT_MULT,RHSMatrixFree,ierr)
        call TSSetRHSJacobian(ts,A,A,PETSC_NULL,PETSC_NULL,ierr)   
      else if (problem .eq. nonlinear) then
C    
C         The user provides the RHS and Jacobian
C    
        call TSSetRHSFunction(ts,RHSFunctionHeat,PETSC_NULL,ierr) 
        call MatCreate(PETSC_COMM_WORLD,M,M,A,ierr) 
        call RHSMatrixHeat(ts,0.d0,A,A,A_structure,PETSC_NULL,ierr)  
        call TSSetRHSJacobian(ts,A,A,RHSJacobianHeat,PETSC_NULL,ierr)   
      endif

      call TSSetFromOptions(ts,ierr)
      call TSGetType(ts,type,PETSC_NULL_CHARACTER,ierr) 

      call TSSetInitialTimeStep(ts,0.d0,dt,ierr) 
      call TSSetDuration(ts,time_steps,100.,ierr) 
      call TSSetSolution(ts,global,ierr) 

      call TSSetUp(ts,ierr) 
      call TSStep(ts,steps,ftime,ierr) 
      call ViewerStringOpen(PETSC_COMM_WORLD,tsinfo,60,viewer,ierr) 
      call TSView(ts,viewer,ierr) 

      call OptionsHasName(PETSC_NULL_CHARACTER,'-test',flg,ierr) 
      if (flg .eq. 1) then
        if (type .eq. TS_EULER) then
          if (abs(nrm_2/steps - 0.00257244) .gt. 1.e-4) then
            print*,'Error in Euler method: 2-norm ',nrm_2/steps,
     &            ' expecting: ',0.00257244
          endif
        else
          if (abs(nrm_2/steps - 0.00506174) .gt. 1.e-4) then
            print*,'Error in ',tsinfo, ': 2-norm ',nrm_2/steps,
     &             ' expecting: ',0.00506174
          endif
        endif
      else 
        print*,size,' Procs Avg. error 2 norm ',nrm_2/steps,
     &              nrm_max/steps,tsinfo
      endif

      call ViewerDestroy(viewer,ierr) 
      call TSDestroy(ts,ierr) 
      call DADestroy(da,ierr) 
      call ViewerDestroy(viewer1,ierr) 
      call ViewerDestroy(viewer2,ierr) 
      call VecDestroy(localwork,ierr) 
      call VecDestroy(csolution,ierr)
      call VecDestroy(local,ierr) 
      call VecDestroy(global,ierr) 
      if (A .ne. 0) then
        call MatDestroy(A,ierr) 
      endif

      call PetscFinalize(ierr)
      stop
      end

C  -------------------------------------------------------------------
 
      subroutine Initial(global,ctx)
#include "FINCLUDE/petsc.h"
#include "FINCLUDE/vec.h"
#include "FINCLUDE/mat.h"
#include "FINCLUDE/da.h"
#include "FINCLUDE/sys.h"
#include "FINCLUDE/sles.h"
#include "FINCLUDE/snes.h"
#include "FINCLUDE/ts.h"
#include "FINCLUDE/viewer.h"
      Vec     global
      integer ctx

      Scalar  localptr(1)
      integer i,mybase,myend,ierr,idx

C
C    Application context data, stored in common block
C
      Vec            localwork,csolution
      DA             da
      Viewer         viewer1,viewer2
      integer        M
      Double         h,nrm_2,nrm_max,nox
      common /tsctx/ localwork,csolution,da,viewer1,viewer2,M,h,nrm_2,
     &               nrm_max,nox

      Double    PETSC_PI
      parameter (PETSC_PI = 3.14159265358979)

C   determine starting point of each processor 

      call VecGetOwnershipRange(global,mybase,myend,ierr) 

C   Initialize the array 

      call VecGetArray(global,localptr,idx,ierr) 
      do 10, i=mybase,myend-1
        localptr(i-mybase+1+idx) = sin(PETSC_PI*i*6.*h) + 
     &                             3.*sin(PETSC_PI*i*2.*h)
 10   continue
      call VecRestoreArray(global,localptr,idx,ierr) 
      return
      end

C ------------------------------------------------------------------------------
C 
C       Exact solution 
C  
      subroutine Solution(t,sol,ctx)
#include "FINCLUDE/petsc.h"
#include "FINCLUDE/vec.h"
#include "FINCLUDE/mat.h"
#include "FINCLUDE/da.h"
#include "FINCLUDE/sys.h"
#include "FINCLUDE/sles.h"
#include "FINCLUDE/snes.h"
#include "FINCLUDE/ts.h"
#include "FINCLUDE/viewer.h"
      Double  t
      Vec     sol
      integer ctx

      Scalar  localptr(1),ex1,ex2,sc1,sc2
      integer i,mybase,myend,ierr,idx

C
C    Application context data, stored in common block
C
      Vec            localwork,csolution
      DA             da
      Viewer         viewer1,viewer2
      integer        M
      Double         h,nrm_2,nrm_max,nox
      common /tsctx/ localwork,csolution,da,viewer1,viewer2,M,h,nrm_2,
     &               nrm_max,nox

      Double    PETSC_PI
      parameter (PETSC_PI = 3.14159265358979)

C   determine starting point of each processor 

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


C -----------------------------------------------------------------------------------

      subroutine Monitor(ts,step,time,global,ctx)
#include "FINCLUDE/petsc.h"
#include "FINCLUDE/vec.h"
#include "FINCLUDE/mat.h"
#include "FINCLUDE/da.h"
#include "FINCLUDE/sys.h"
#include "FINCLUDE/sles.h"
#include "FINCLUDE/snes.h"
#include "FINCLUDE/ts.h"
#include "FINCLUDE/draw.h"
#include "FINCLUDE/viewer.h"
      TS      ts
      integer step,ctx,ierr
      Double  time,lnrm_2,lnrm_max
      Vec     global
      Scalar  mone

C
C    Application context data, stored in common block
C
      Vec            localwork,csolution
      DA             da
      Viewer         viewer1,viewer2
      integer        M
      Double         h,nrm_2,nrm_max,nox
      common /tsctx/ localwork,csolution,da,viewer1,viewer2,M,h,nrm_2,
     &               nrm_max,nox


      call VecView(global,viewer1,ierr) 

      call Solution(time,csolution, ctx) 
      mone = -1.0
      call VecAXPY(mone,global,csolution,ierr) 
      call VecNorm(csolution,NORM_2,lnrm_2,ierr) 
      lnrm_2 = sqrt(h)*lnrm_2
      call VecNorm(csolution,NORM_MAX,lnrm_max,ierr) 

      if (nox .eq. 0) then
        print*,'Time-step ',step,' time ',time,' norm of error ',
     &         lnrm_2,lnrm_max
      endif

      nrm_2   = nrm_2 + lnrm_2
      nrm_max = nrm_max +  lnrm_max

      call VecView(csolution,viewer2,ierr) 

      return   
      end

C  -----------------------------------------------------------------------

      subroutine RHSMatrixFree(mat,x,y,ierr)
#include "FINCLUDE/petsc.h"
#include "FINCLUDE/vec.h"
#include "FINCLUDE/mat.h"
#include "FINCLUDE/da.h"
#include "FINCLUDE/sys.h"
#include "FINCLUDE/sles.h"
#include "FINCLUDE/snes.h"
#include "FINCLUDE/ts.h"
#include "FINCLUDE/viewer.h"
      Mat     mat
      Vec     x,y
      integer ierr

      call RHSFunctionHeat(0,0.0,x,y,0,ierr) 
      return
      end

C -------------------------------------------------------------------------

      subroutine RHSFunctionHeat(ts,t,globalin,globalout,ctx,ierr)
#include "FINCLUDE/petsc.h"
#include "FINCLUDE/vec.h"
#include "FINCLUDE/mat.h"
#include "FINCLUDE/da.h"
#include "FINCLUDE/sys.h"
#include "FINCLUDE/sles.h"
#include "FINCLUDE/snes.h"
#include "FINCLUDE/ts.h"
#include "FINCLUDE/viewer.h"
      TS      ts
      Double  t
      Vec     globalin, globalout
      integer ctx,ierr
      DA      da
      Vec     local
      integer ierr,i,localsize,cdx,ldx 
      Scalar  copyptr(1), localptr(1),sc

C
C    Application context data, stored in common block
C
      Vec            localwork,csolution
      DA             da
      Viewer         viewer1,viewer2
      integer        M
      Double         h,nrm_2,nrm_max,nox
      common /tsctx/ localwork,csolution,da,viewer1,viewer2,M,h,nrm_2,
     &               nrm_max,nox

C  Extract local array  

      call DAGetLocalVector(da,local,ierr) 
      call DAGlobalToLocalBegin(da,globalin,INSERT_VALUES,local,ierr) 
      call DAGlobalToLocalEnd(da,globalin,INSERT_VALUES,local,ierr) 
      call VecGetArray(local,localptr,ldx,ierr) 

C  Extract work vector 

      call VecGetArray(localwork,copyptr,cdx,ierr) 

C   Update Locally - Make array of new values 
C   Note: For the first and last entry I copy the value 
C   if this is an interior node it is irrelevant 

      sc = 1.0/(h*h)
      call VecGetLocalSize(local,localsize,ierr) 
      copyptr(1+cdx) = localptr(1+ldx)
      do 10, i=1,localsize-2
        copyptr(i+1+cdx) = sc * (localptr(i+2+ldx) + localptr(i+ldx) - 
     &                     2.0*localptr(i+1+ldx))
 10   continue
      copyptr(localsize-1+1+cdx) = localptr(localsize-1+1+ldx)
      call VecRestoreArray(localwork,copyptr,cdx,ierr) 

C   Local to Global 

      call DALocalToGlobal(da,localwork,INSERT_VALUES,globalout,ierr) 
      return 
      end

C  ---------------------------------------------------------------------

      subroutine RHSMatrixHeat(ts,t,AA,BB,str,ctx,ierr)
#include "FINCLUDE/petsc.h"
#include "FINCLUDE/vec.h"
#include "FINCLUDE/mat.h"
#include "FINCLUDE/da.h"
#include "FINCLUDE/sys.h"
#include "FINCLUDE/pc.h"
#include "FINCLUDE/sles.h"
#include "FINCLUDE/snes.h"
#include "FINCLUDE/ts.h"
#include "FINCLUDE/viewer.h"
      Mat          AA,BB
      Double       t
      TS           ts
      MatStructure str
      integer      ctx,ierr

      Mat          A
      integer      i,mstart,mend,rank,size, idx(3)
      Scalar       v(3),stwo, sone

C
C    Application context data, stored in common block
C
      Vec            localwork,csolution
      DA             da
      Viewer         viewer1,viewer2
      integer        M
      Double         h,nrm_2,nrm_max,nox
      common /tsctx/ localwork,csolution,da,viewer1,viewer2,M,h,nrm_2,
     &               nrm_max,nox

      A    = AA
      stwo = -2./(h*h)
      sone = -.5*stwo
      str  = SAME_NONZERO_PATTERN

      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)

      call MatGetOwnershipRange(A,mstart,mend,ierr) 
      if (mstart .eq. 0) then
        v(1) = 1.0
        call MatSetValues(A,1,mstart,1,mstart,v,INSERT_VALUES,ierr) 
        mstart = mstart + 1
      endif
      if (mend .eq. M) then
        mend = mend - 1
        v(1) = 1.0
        call MatSetValues(A,1,mend,1,mend,v,INSERT_VALUES,ierr) 
      endif

C  
C     Construct matrice one row at a time
C  
      v(1) = sone
      v(2) = stwo
      v(3) = sone  
      do 10, i=mstart,mend-1
        idx(1) = i-1
        idx(2) = i 
        idx(3) = i+1
        call MatSetValues(A,1,i,3,idx,v,INSERT_VALUES,ierr) 
 10   continue

      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr) 
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr) 
      return
      end

C --------------------------------------------------------------------------------------

      subroutine RHSJacobianHeat(ts,t,x,AA,BB,str,ctx,ierr)
#include "FINCLUDE/petsc.h"
#include "FINCLUDE/vec.h"
#include "FINCLUDE/mat.h"
#include "FINCLUDE/da.h"
#include "FINCLUDE/sys.h"
#include "FINCLUDE/sles.h"
#include "FINCLUDE/snes.h"
#include "FINCLUDE/ts.h"
#include "FINCLUDE/viewer.h"
      TS           ts
      Double       t
      Vec          x
      Mat          AA,BB
      MatStructure str
      integer      ctx,ierr

      call RHSMatrixHeat(ts,t,AA,BB,str,ctx,ierr)
      return
      end






