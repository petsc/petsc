C
C "$Id: ex1f.F,v 1.1 1996/09/30 21:29:20 bsmith Exp bsmith $"
C
C    Solves the time dependent Bratu problem using pseudo-timestepping
C
C
C   Concepts: TS,Pseudo-timestepping,nonlinear problems
C   Routines: TSCreate(),TSSetSolution(),TSSetRHSFunction(),TSSetRHSJacobian()
C   Routines: TSSetType(), TS_PSEUDO, TSSetInitialTimeStep(),TSSetDuration()
C   Routines: TSSetFromOptions(),TSStep(),TSDestroy()
C   Processors: 1
C
C     Demonstrates how one may solve a nonlinear problem 
C   with pseudo timestepping. In this simple example, the pseudo-time
C   step is the same for all grid points, i.e. this is equivalent to 
C   a backward Euler with variable timestep.
C
C     Note: this example does not require pseudo-timestepping since it
C   is an easy nonlinear problem; but it is included to demonstrate how
C   the pseudo-timestepping may be done.
C
C     See snes/examples/tutorials/ex4.c[ex4f.F] and 
C   snes/examples/tutorials/ex5.c[ex5f.F] where the problem is described
C   and solved using Newton's method alone.
C
C    Include "ts.h"    to use the PETSc timestepping routines.
C            "petsc.h" for basic PETSc operations
C            "mat.h"   for matrix operations
C            "vec.h"   for vector operations
C
C23456789012345678901234567890123456789012345678901234567890123456789012
      program main
      implicit none
#include "FINCLUDE/petsc.h"
#include "FINCLUDE/vec.h"
#include "FINCLUDE/mat.h"
#include "FINCLUDE/pc.h"
#include "FINCLUDE/ts.h"
C
C  Create an application context to contain data needed by the 
C  application-provided call-back routines, FormJacobian() and
C  FormFunction(). We use a double precision array with three
C  entries indexed by param, lmx,lmy 
C
      Double    user(3)
      integer   param,lmx,lmy
      parameter (param = 1,lmx = 2,lmy = 3)
C
C   User-defined routines
C
      external FormJacobian,FormFunction
C
C   Data for problem
C
      TS       ts                
      Vec      x, r              
      Mat      J                 
      integer  its               
      integer  ierr, N, flg 
      Double   param_max, param_min, dt
      Double   ftime

      param_max = 6.81
      param_min = 0
      dt = 1.d-6

      call PetscInitialize( PETSC_NULL_CHARACTER ,ierr)
      user(lmx)        = 4
      user(lmy)        = 4
      user(param)      = 6.0
  
C
C     Allow user to set the grid dimensions and nonlinearity parameter at run-time
C
      call OptionsGetDouble(PETSC_NULL_CHARACTER,'-mx',user(lmx),flg,
     &                      ierr)
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-my',user(lmy),flg,ierr)
      call OptionsGetDouble(PETSC_NULL_CHARACTER,'-param',user(param),
     &                      flg,ierr)
      if (user(param) .ge. param_max .or. 
     &                                user(param) .le. param_min) then
        print*,'Parameter is out of range'
      endif
      call OptionsGetDouble(PETSC_NULL_CHARACTER,'-dt',dt,flg,ierr)
      N          = user(lmx)*user(lmy)
  
C
C      Create vectors to hold the solution and function value
C
       call VecCreateSeq(MPI_COMM_SELF,N,x,ierr)
       call VecDuplicate(x,r,ierr) 

C
C    Create matrix to hold Jacobian. Preallocate 5 nonzeros per row
C    in the sparse matrix. Note that this is not the optimal strategy see
C    the Performance chapter of the users manual for information on 
C    preallocating memory in sparse matrices.
C

      call MatCreateSeqAIJ(MPI_COMM_SELF,N,N,5,PETSC_NULL,J,ierr) 

C
C     Create timestepper context 
C

      call TSCreate(MPI_COMM_WORLD,TS_NONLINEAR,ts,ierr) 

C
C     Tell the timestepper context where to compute solutions
C

      call TSSetSolution(ts,x,ierr) 

C
C    Provide the call-back for the nonlinear function we are 
C    evaluating. Thus whenever the timestepping routines need the
C    function they will call this routine. Note the final argument
C    is the application context used by the call-back functions.
C

      call TSSetRHSFunction(ts,FormFunction,user,ierr) 

C
C     Set the Jacobian matrix and the function used to compute 
C     Jacobians.
C

      call TSSetRHSJacobian(ts,J,J,FormJacobian,user,ierr)

C
C       For the initial guess for the problem
C

      call FormInitialGuess(x,user,ierr)

C
C       This indicates that we are using pseudo timestepping to 
C     find a steady state solution to the nonlinear problem.
C

      call TSSetType(ts,TS_PSEUDO,ierr) 

C
C       Set the initial time to start at (this is arbitrary for 
C     steady state problems and the initial timestep given above
C

      call TSSetInitialTimeStep(ts,0.d0,dt,ierr) 

C
C      Set a large number of timesteps and final duration time
C     to insure convergence to steady state.
C

      call TSSetDuration(ts,1000,1.d12,ierr)

C
C      Set any additional options from the options database. This
C     includes all options for the nonlinear and linear solvers used
C     internally the the timestepping routines.
C

      call TSSetFromOptions(ts,ierr) 

      call TSSetUp(ts,ierr) 

C
C      Perform the solve. This is where the timestepping takes place.
C
 
      call TSStep(ts,its,ftime,ierr) 
   
      write(6,100) its,ftime
 100  format('Number of pseudo timesteps ',i5,' final time ',e8.2)

C
C     Free the data structures constructed above
C

      call VecDestroy(x,ierr) 
      call VecDestroy(r,ierr) 
      call MatDestroy(J,ierr) 
      call TSDestroy(ts,ierr) 
      call PetscFinalize(ierr)
      stop
      end

C
C  --------------------  Form initial approximation ----------------- 
C
      subroutine FormInitialGuess(X,user,ierr)
      implicit none
#include "FINCLUDE/petsc.h"
#include "FINCLUDE/vec.h"
#include "FINCLUDE/mat.h"
#include "FINCLUDE/pc.h"
#include "FINCLUDE/ts.h"
      Vec       X
      Double    user(3)
      integer   i, j, row, mx, my, ierr,xidx
      Double    two, one, lambda
      Double    temp1, temp, hx, hy, hxdhy, hydhx
      Double    sc
      Scalar    xx(1)
      integer   param,lmx,lmy
      parameter (param = 1,lmx = 2,lmy = 3)

      two = 2.0
      one = 1.0

      mx     = user(lmx) 
      my     = user(lmy)
      lambda = user(param)

      hy    = one / (my-1)
      hx    = one / (mx-1)
      sc    = hx*hy
      hxdhy = hx/hy
      hydhx = hy/hx

      call VecGetArray(X,xx,xidx,ierr) 
      temp1 = lambda/(lambda + one)
      do 10, j=1,my
        temp = DBLE(min(j-1,my-j))*hy
        do 20 i=1,mx
          row = i + (j-1)*mx  
          if (i .eq. 1 .or. j .eq. 1 .or. 
     &        i .eq. mx .or. j .eq. my ) then
            xx(row+xidx) = 0.0 
          else
            xx(row+xidx) = 
     &        temp1*sqrt(min(DBLE(min(i-1,mx-i))*hx,temp)) 
          endif
 20     continue
 10   continue
      call VecRestoreArray(X,xx,xidx,ierr) 
      return
      end
C
C  --------------------  Evaluate Function F(x) --------------------- 
C
      subroutine FormFunction(ts,t,X,F,user,ierr)
      implicit none
#include "FINCLUDE/petsc.h"
#include "FINCLUDE/vec.h"
#include "FINCLUDE/mat.h"
#include "FINCLUDE/pc.h"
#include "FINCLUDE/ts.h"
      TS       ts
      Double   t
      Vec      X,F
      Double   user(3)
      integer  ierr, i, j, row, mx, my,xidx,fidx
      Double   two, one, lambda
      Double   hx, hy, hxdhy, hydhx
      Scalar   ut, ub, ul, ur, u, uxx, uyy, sc,xx(1),ff(1)

      integer   param,lmx,lmy
      parameter (param = 1,lmx = 2,lmy = 3)

      two = 2.0
      one = 1.0

      mx     = user(lmx) 
      my     = user(lmy)
      lambda = user(param)

      hx    = 1.0 / DBLE(mx-1)
      hy    = 1.0 / DBLE(my-1)
      sc    = hx*hy
      hxdhy = hx/hy
      hydhx = hy/hx

      call VecGetArray(X,xx,xidx,ierr) 
      call VecGetArray(F,ff,fidx,ierr) 
      do 10 j=1,my
        do 20 i=1,mx
          row = i + (j-1)*mx
          if (i .eq. 1 .or. j .eq. 1 .or. 
     &        i .eq. mx .or. j .eq. my ) then
            ff(row+fidx) = xx(row+xidx)
          else
            u            = xx(row + xidx)
            ub           = xx(row - mx + xidx)
            ul           = xx(row - 1 + xidx)
            ut           = xx(row + mx + xidx)
            ur           = xx(row + 1 + xidx)
            uxx          = (-ur + two*u - ul)*hydhx
            uyy          = (-ut + two*u - ub)*hxdhy
            ff(row+fidx) = -uxx - uyy + sc*lambda*exp(u)
            u =  -uxx - uyy + sc*lambda*exp(u)
         endif
 20   continue
 10   continue

      call VecRestoreArray(X,xx,xidx,ierr) 
      call VecRestoreArray(F,ff,fidx,ierr) 
      return 
      end
C
C  --------------------  Evaluate Jacobian F'(x) -------------------- 
C
      subroutine FormJacobian(ts,ctime,X,JJ,B,flag,user,ierr)
      implicit none
#include "FINCLUDE/petsc.h"
#include "FINCLUDE/vec.h"
#include "FINCLUDE/mat.h"
#include "FINCLUDE/pc.h"
#include "FINCLUDE/ts.h"
      TS           ts
      Vec          X
      Mat          JJ,B
      MatStructure flag
      Double       user(3),ctime
      Mat          jac
      integer      i, j, row, mx, my, col(5), ierr,xidx
      Scalar       two, one, lambda, v(5), sc, xx(1)
      Double       hx, hy, hxdhy, hydhx

      integer   param,lmx,lmy
      parameter (param = 1,lmx = 2,lmy = 3)

      jac = B
      two = 2.0
      one = 1.0

      mx     = user(lmx) 
      my     = user(lmy)
      lambda = user(param)

      hx    = 1.0 / DBLE(mx-1)
      hy    = 1.0 / DBLE(my-1)
      sc    = hx*hy
      hxdhy = hx/hy
      hydhx = hy/hx

      call VecGetArray(X,xx,xidx,ierr) 
      do 10 j=1,my
        do 20 i=1,mx
C
C      When inserting into PETSc matrices, indices start at 0
C
C       call PetscTrValid(ierr)
          row = i - 1 + (j-1)*mx
          if (i .eq. 1 .or. j .eq. 1 .or. 
     &        i .eq. mx .or. j .eq. my ) then
            call MatSetValues(jac,1,row,1,row,one,INSERT_VALUES,ierr) 
          else
            v(1)   = hxdhy 
            col(1) = row - mx
            v(2)   = hydhx 
            col(2) = row - 1
            v(3)   = -two*(hydhx+hxdhy)+sc*lambda*exp(xx(row+1+xidx)) 
            col(3) = row
            v(4)   = hydhx 
            col(4) = row + 1
            v(5)   = hxdhy 
            col(5) = row + mx
            call MatSetValues(jac,1,row,5,col,v,INSERT_VALUES,ierr) 
          endif
 20     continue
 10   continue
      call MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr) 
      call MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr)
      call VecRestoreArray(X,xx,xidx,ierr) 
      flag = SAME_NONZERO_PATTERN
C       call PetscTrValid(ierr)
      return
      end




