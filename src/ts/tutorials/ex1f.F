!
!   Solves the time dependent Bratu problem using pseudo-timestepping
!
!   Concepts: TS^pseudo-timestepping
!   Concepts: pseudo-timestepping
!   Concepts: TS^nonlinear problems
!   Processors: 1
!
!   This code demonstrates how one may solve a nonlinear problem
!   with pseudo-timestepping. In this simple example, the pseudo-timestep
!   is the same for all grid points, i.e., this is equivalent to using
!   the backward Euler method with a variable timestep.
!
!   Note: This example does not require pseudo-timestepping since it
!   is an easy nonlinear problem, but it is included to demonstrate how
!   the pseudo-timestepping may be done.
!
!   See snes/tutorials/ex4.c[ex4f.F] and
!   snes/tutorials/ex5.c[ex5f.F] where the problem is described
!   and solved using the method of Newton alone.
!
!
!23456789012345678901234567890123456789012345678901234567890123456789012
      program main
#include <petsc/finclude/petscts.h>
      use petscts
      implicit none

!
!  Create an application context to contain data needed by the
!  application-provided call-back routines, FormJacobian() and
!  FormFunction(). We use a double precision array with three
!  entries indexed by param, lmx, lmy.
!
      PetscReal user(3)
      PetscInt          param,lmx,lmy,i5
      parameter (param = 1,lmx = 2,lmy = 3)
!
!   User-defined routines
!
      external FormJacobian,FormFunction
!
!   Data for problem
!
      TS                ts
      Vec               x,r
      Mat               J
      PetscInt           its,N,i1000,itmp
      PetscBool  flg
      PetscErrorCode      ierr
      PetscReal  param_max,param_min,dt
      PetscReal  tmax
      PetscReal  ftime
      TSConvergedReason reason

      i5 = 5
      param_max = 6.81
      param_min = 0

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
         print*,'Unable to initialize PETSc'
         stop
      endif
      user(lmx)        = 4
      user(lmy)        = 4
      user(param)      = 6.0

!
!     Allow user to set the grid dimensions and nonlinearity parameter at run-time
!
      call PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,                    &
     &     '-mx',user(lmx),flg,ierr)
      itmp = 4
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,            &
     &                        '-my',itmp,flg,ierr)
      user(lmy) = itmp
      call PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,           &
     &                        '-param',user(param),flg,ierr)
      if (user(param) .ge. param_max .or.                               &
     &                                user(param) .le. param_min) then
        print*,'Parameter is out of range'
      endif
      if (user(lmx) .gt. user(lmy)) then
        dt = .5/user(lmx)
      else
        dt = .5/user(lmy)
      endif
      call PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,       &
     &                         '-dt',dt,flg,ierr)
      N          = int(user(lmx)*user(lmy))

!
!      Create vectors to hold the solution and function value
!
      call VecCreateSeq(PETSC_COMM_SELF,N,x,ierr)
      call VecDuplicate(x,r,ierr)

!
!    Create matrix to hold Jacobian. Preallocate 5 nonzeros per row
!    in the sparse matrix. Note that this is not the optimal strategy see
!    the Performance chapter of the users manual for information on
!    preallocating memory in sparse matrices.
!
      i5 = 5
      call MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,i5,PETSC_NULL_INTEGER,    &
     &     J,ierr)

!
!     Create timestepper context
!

      call TSCreate(PETSC_COMM_WORLD,ts,ierr)
      call TSSetProblemType(ts,TS_NONLINEAR,ierr)

!
!     Tell the timestepper context where to compute solutions
!

      call TSSetSolution(ts,x,ierr)

!
!    Provide the call-back for the nonlinear function we are
!    evaluating. Thus whenever the timestepping routines need the
!    function they will call this routine. Note the final argument
!    is the application context used by the call-back functions.
!

      call TSSetRHSFunction(ts,PETSC_NULL_VEC,FormFunction,user,ierr)

!
!     Set the Jacobian matrix and the function used to compute
!     Jacobians.
!

      call TSSetRHSJacobian(ts,J,J,FormJacobian,user,ierr)

!
!       For the initial guess for the problem
!

      call FormInitialGuess(x,user,ierr)

!
!       This indicates that we are using pseudo timestepping to
!     find a steady state solution to the nonlinear problem.
!

      call TSSetType(ts,TSPSEUDO,ierr)

!
!       Set the initial time to start at (this is arbitrary for
!     steady state problems and the initial timestep given above
!

      call TSSetTimeStep(ts,dt,ierr)

!
!      Set a large number of timesteps and final duration time
!     to insure convergence to steady state.
!
      i1000 = 1000
      tmax  = 1.e12
      call TSSetMaxSteps(ts,i1000,ierr)
      call TSSetMaxTime(ts,tmax,ierr)
      call TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER,ierr)

!
!      Set any additional options from the options database. This
!     includes all options for the nonlinear and linear solvers used
!     internally the timestepping routines.
!

      call TSSetFromOptions(ts,ierr)

      call TSSetUp(ts,ierr)

!
!      Perform the solve. This is where the timestepping takes place.
!
      call TSSolve(ts,x,ierr)
      call TSGetSolveTime(ts,ftime,ierr)
      call TSGetStepNumber(ts,its,ierr)
      call TSGetConvergedReason(ts,reason,ierr)

      write(6,100) its,ftime,reason
 100  format('Number of pseudo time-steps ',i5,' final time ',1pe9.2    &
     &     ,' reason ',i3)

!
!     Free the data structures constructed above
!

      call VecDestroy(x,ierr)
      call VecDestroy(r,ierr)
      call MatDestroy(J,ierr)
      call TSDestroy(ts,ierr)
      call PetscFinalize(ierr)
      end

!
!  --------------------  Form initial approximation -----------------
!
      subroutine FormInitialGuess(X,user,ierr)
      use petscts
      implicit none

      Vec              X
      PetscReal user(3)
      PetscInt  i,j,row,mx,my
      PetscErrorCode ierr
      PetscOffset      xidx
      PetscReal one,lambda
      PetscReal temp1,temp,hx,hy
      PetscScalar      xx(2)
      PetscInt          param,lmx,lmy
      parameter (param = 1,lmx = 2,lmy = 3)

      one = 1.0

      mx     = int(user(lmx))
      my     = int(user(lmy))
      lambda = user(param)

      hy    = one / (my-1)
      hx    = one / (mx-1)

      call VecGetArray(X,xx,xidx,ierr)
      temp1 = lambda/(lambda + one)
      do 10, j=1,my
        temp = min(j-1,my-j)*hy
        do 20 i=1,mx
          row = i + (j-1)*mx
          if (i .eq. 1 .or. j .eq. 1 .or.                               &
     &        i .eq. mx .or. j .eq. my) then
            xx(row+xidx) = 0.0
          else
            xx(row+xidx) =                                              &
     &        temp1*sqrt(min(min(i-1,mx-i)*hx,temp))
          endif
 20     continue
 10   continue
      call VecRestoreArray(X,xx,xidx,ierr)
      return
      end
!
!  --------------------  Evaluate Function F(x) ---------------------
!
      subroutine FormFunction(ts,t,X,F,user,ierr)
      use petscts
      implicit none

      TS       ts
      PetscReal  t
      Vec               X,F
      PetscReal  user(3)
      PetscErrorCode     ierr
      PetscInt         i,j,row,mx,my
      PetscOffset       xidx,fidx
      PetscReal  two,lambda
      PetscReal  hx,hy,hxdhy,hydhx
      PetscScalar  ut,ub,ul,ur,u
      PetscScalar  uxx,uyy,sc
      PetscScalar  xx(2),ff(2)
      PetscInt     param,lmx,lmy
      parameter (param = 1,lmx = 2,lmy = 3)

      two = 2.0

      mx     = int(user(lmx))
      my     = int(user(lmy))
      lambda = user(param)

      hx    = 1.0 / real(mx-1)
      hy    = 1.0 / real(my-1)
      sc    = hx*hy
      hxdhy = hx/hy
      hydhx = hy/hx

      call VecGetArrayRead(X,xx,xidx,ierr)
      call VecGetArray(F,ff,fidx,ierr)
      do 10 j=1,my
        do 20 i=1,mx
          row = i + (j-1)*mx
          if (i .eq. 1 .or. j .eq. 1 .or.                               &
     &        i .eq. mx .or. j .eq. my) then
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

      call VecRestoreArrayRead(X,xx,xidx,ierr)
      call VecRestoreArray(F,ff,fidx,ierr)
      return
      end
!
!  --------------------  Evaluate Jacobian of F(x) --------------------
!
      subroutine FormJacobian(ts,ctime,X,JJ,B,user,ierr)
      use petscts
      implicit none

      TS               ts
      Vec              X
      Mat              JJ,B
      PetscReal user(3),ctime
      PetscErrorCode   ierr
      Mat              jac
      PetscOffset xidx
      PetscInt    i,j,row(1),mx,my
      PetscInt    col(5),i1,i5
      PetscScalar two,one,lambda
      PetscScalar v(5),sc,xx(2)
      PetscReal hx,hy,hxdhy,hydhx

      PetscInt  param,lmx,lmy
      parameter (param = 1,lmx = 2,lmy = 3)

      i1 = 1
      i5 = 5
      jac = B
      two = 2.0
      one = 1.0

      mx     = int(user(lmx))
      my     = int(user(lmy))
      lambda = user(param)

      hx    = 1.0 / real(mx-1)
      hy    = 1.0 / real(my-1)
      sc    = hx*hy
      hxdhy = hx/hy
      hydhx = hy/hx

      call VecGetArrayRead(X,xx,xidx,ierr)
      do 10 j=1,my
        do 20 i=1,mx
!
!      When inserting into PETSc matrices, indices start at 0
!
          row(1) = i - 1 + (j-1)*mx
          if (i .eq. 1 .or. j .eq. 1 .or.                               &
     &        i .eq. mx .or. j .eq. my) then
            call MatSetValues(jac,i1,row,i1,row,one,INSERT_VALUES,ierr)
          else
            v(1)   = hxdhy
            col(1) = row(1) - mx
            v(2)   = hydhx
            col(2) = row(1) - 1
            v(3)   = -two*(hydhx+hxdhy)+sc*lambda*exp(xx(row(1)+1+xidx))
            col(3) = row(1)
            v(4)   = hydhx
            col(4) = row(1) + 1
            v(5)   = hxdhy
            col(5) = row(1) + mx
            call MatSetValues(jac,i1,row,i5,col,v,INSERT_VALUES,ierr)
          endif
 20     continue
 10   continue
      call MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr)
      call VecRestoreArray(X,xx,xidx,ierr)
      return
      end

!/*TEST
!
!    test:
!      TODO: broken
!      args: -ksp_gmres_cgs_refinement_type refine_always -snes_type newtonls -ts_monitor_pseudo -ts_max_snes_failures 3 -ts_pseudo_frtol 1.e-5  -snes_stol 1e-5
!
!TEST*/


