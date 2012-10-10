
!
!/*T
!   Concepts: TS^time-dependent nonlinear problems
!   Processors: n
!T*/
!
!  ------------------------------------------------------------------------
!
!   This program solves a simple time-dependent nonlinear PDE using implicit
!   timestepping:
!                                    u * u_xx
!                              u_t = ---------
!                                    2*(t+1)^2
!
!             u(0,x) = 1 + x*x; u(t,0) = t + 1; u(t,1) = 2*t + 2
!
!   The exact solution is u(t,x) = (1 + x*x) * (1 + t).
!
!   Note that since the solution is linear in time and quadratic in x,
!   the finite difference scheme actually computes the "exact" solution.
!
!   We use the backward Euler method.
!
!  --------------------------------------------------------------------------

      program main
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                    Include files
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Each routine within this program uses the include file 'ex2f.h',
!  which itself includes the various PETSc include files as well as
!  problem-specific data in several common blocks.
!
!  This program uses CPP for preprocessing, as indicated by the use of
!  PETSc include files in the directory petsc/include/finclude.  This
!  convention enables use of the CPP preprocessor, which allows the use
!  of the #include statements that define PETSc objects and variables.
!
!  Use of the conventional Fortran include statements is also supported
!  In this case, the PETsc include files are located in the directory
!  petsc/include/foldinclude.
!
!  Since one must be very careful to include each file no more than once
!  in a Fortran routine, application programmers must exlicitly list
!  each file needed for the various PETSc components within their
!  program (unlike the C/C++ interface).
!
!  See the Fortran section of the PETSc users manual for details.

#include "ex2f.h"

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Variables:
!     ts             - timestepping solver
!     A              - Jacobian matrix context
!     u_local        - local vector
!     u              - approximate solution vector
!     ftime          - final time
!     time_total_max - default total max time
!     time_steps_max - default max timesteps
!
!  Note that vectors are declared as PETSc "Vec" objects.  These vectors
!  are mathematical objects that contain more than just an array of
!  double precision numbers. I.e., vectors in PETSc are not just
!        double precision x(*).
!  However, local vector data can be easily accessed via VecGetArray().
!  See the Fortran section of the PETSc users manual for details.

      TS               ts
      Vec              u
      Mat              A
      PetscBool        flg,fdjac,monitor
      PetscErrorCode ierr
      PetscInt  time_steps_max,i1,steps
      PetscReal dt,time_total_max,ftime,norm

!  Note: Any user-defined Fortran routines (such as RHSFunction)
!  MUST be declared as external.

      external MyMonitor,RHSFunction
      external InitialConditions,RHSJacobian

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      comm = PETSC_COMM_WORLD
      call MPI_Comm_size(comm,size,ierr)
      call MPI_Comm_rank(comm,rank,ierr)

!  Initialize problem parameters

      time_steps_max = 1000
      time_total_max = 100.0
      m              = 60
      debug          = .false.
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-m',M,flg,ierr)
      call PetscOptionsGetBool(PETSC_NULL_CHARACTER,'-debug',debug,flg, &
     &     ierr)
      call PetscOptionsGetBool(PETSC_NULL_CHARACTER,'-monitor',monitor, &
     &                         flg,ierr)
      one_d0  = 1.0d0
      two_d0  = 2.0d0
      four_d0 = 4.0d0
      h       = one_d0/(m-one_d0)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create vector data structures
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create distributed array (DMDA) to manage parallel grid and vectors
!  Set up the ghost point communication pattern.  There are m total
!  grid values spread equally among all the processors.
      i1 = 1
      call DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,m,i1,i1,    &
     &     PETSC_NULL_INTEGER,da,ierr)

!  Extract global and local vectors from DMDA; then duplicate for remaining
!  vectors that are the same types.

      call DMCreateGlobalVector(da,u,ierr)
      call DMCreateLocalVector(da,u_local,ierr)

!  Make local work vector for evaluating right-hand-side function
      call VecDuplicate(u_local,localwork,ierr)

!  Make global work vector for storing exact solution
      call VecDuplicate(u,solution,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create timestepping solver context; set callback routine for
!  right-hand-side function evaluation.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call TSCreate(comm,ts,ierr)
      call TSSetProblemType(ts,TS_NONLINEAR,ierr)
      call TSSetRHSFunction(ts,PETSC_NULL_OBJECT,RHSFunction,           &
     &                      PETSC_NULL_OBJECT,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Set optional user-defined monitoring routine
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


      if (monitor) then
        call TSMonitorSet(ts,MyMonitor,PETSC_NULL_OBJECT,               &
     &                    PETSC_NULL_FUNCTION,ierr)
      endif

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  For nonlinear problems, the user can provide a Jacobian evaluation
!  routine (or use a finite differencing approximation).
!
!  Create matrix data structure; set Jacobian evaluation routine.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call MatCreate(comm,A,ierr)
      call MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,m,ierr)
      call MatSetFromOptions(A,ierr)
      call MatSetUp(A,ierr)
      call TSSetRHSJacobian(ts,A,A,RHSJacobian,PETSC_NULL_OBJECT,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Set solution vector and initial timestep
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      dt = h/two_d0
      call TSSetInitialTimeStep(ts,0.d0,dt,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Customize timestepping solver:
!     - Set the solution method to be the Backward Euler method.
!     - Set timestepping duration info
!  Then set runtime options, which can override these defaults.
!  For example,
!     -ts_max_steps <maxsteps> -ts_final_time <maxtime>
!  to override the defaults set by TSSetDuration().
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call TSSetType(ts,TSTHETA,ierr)
      call TSThetaSetTheta(ts,one_d0,ierr)
      call TSSetDuration(ts,time_steps_max,time_total_max,ierr)
      call TSSetFromOptions(ts,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Solve the problem
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Evaluate initial conditions

      call InitialConditions(u)

!  Run the timestepping solver

      call TSSolve(ts,u,ftime,ierr)
      call TSGetTimeStepNumber(ts,steps,ierr)
      call VecNorm(u,NORM_2,norm,ierr)

      write(6,100) steps,ftime,norm
 100  format('Number of pseudo time-steps ',i5,' final time ',1pe9.2,   &
     & ' solution norm ',1pe9.2)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call TSDestroy(ts,ierr)
      call VecDestroy(localwork,ierr)
      call VecDestroy(solution,ierr)
      call VecDestroy(u_local,ierr)
      call VecDestroy(u,ierr)
      call DMDestroy(da,ierr)
      call MatDestroy(A,ierr)

!  Always call PetscFinalize() before exiting a program.  This routine
!    - finalizes the PETSc libraries as well as MPI
!    - provides summary and diagnostic information if certain runtime
!      options are chosen (e.g., -log_summary).

      call PetscFinalize(ierr)
      end

!  ------------------------------------------------------------------------
!
!  InitialConditions - Computes the solution at the initial time.
!
!  Input Parameters:
!     u - uninitialized solution vector (global)
!     appctx - user-defined application context
!
!  Output Parameter:
!     u - vector with solution at initial time (global)
!
      subroutine InitialConditions(u)
      implicit none
#include "ex2f.h"

!  Input/output parameters:
      Vec     u

!  Local variables:
      PetscScalar localptr(1),x
      PetscInt    i,mybase,myend
      PetscErrorCode ierr
      PetscOffset idx

!  Determine starting and ending point of each processor's range of
!  grid values.  Note that we shift by 1 to convert from the 0-based
!  C convention of starting indices to the 1-based Fortran convention.

      call VecGetOwnershipRange(u,mybase,myend,ierr)
      mybase = mybase + 1

!  Get a pointer to vector data.
!    - For default PETSc vectors, VecGetArray() returns a pointer to
!      the data array.  Otherwise, the routine is implementation dependent.
!    - You MUST call VecRestoreArray() when you no longer need access to
!      the array.
!    - Note that the Fortran interface to VecGetArray() differs from the
!      C version.  See the users manual for details.

      call VecGetArray(u,localptr,idx,ierr)

!     We initialize the solution array by simply writing the solution
!     directly into the array locations.  Alternatively, we could use
!     VecSetValues() or VecSetValuesLocal().

      do 10, i=mybase,myend
!       x - current location in global grid
        x = h*(i-1)
        localptr(i-mybase+idx+1) = one_d0 + x*x
 10   continue

!  Restore vector

      call VecRestoreArray(u,localptr,idx,ierr)

!  Print debugging information if desired
      if (debug) then
        if (rank .eq. 0) write(6,*) 'initial guess vector'
        call VecView(u,PETSC_VIEWER_STDOUT_WORLD,ierr)
      endif

      return
      end

!  ------------------------------------------------------------------------
!
!  ExactSolution - Computes the exact solution at a given time.
!
!  Input Parameters:
!    t - current time
!    appctx - user-defined application context
!
!  Output Parameter:
!    ierr - error code
!
!  Note: The solution vector is stored in the common block!
!
      subroutine ExactSolution(t,ierr)
      implicit none
#include "ex2f.h"

!  Input/output parameters:
      PetscReal  t
      PetscScalar  x
      PetscErrorCode          ierr

!  Local variables:
      PetscScalar localptr(1)
      PetscInt    i,mybase,myend
      PetscOffset idx

!  Determine starting and ending point of each processors range of
!  grid values.  Note that we shift by 1 to convert from the 0-based
!  C convention of starting indices to the 1-based Fortran convention.

      call VecGetOwnershipRange(solution,mybase,myend,ierr)
      mybase = mybase + 1

!  Get a pointer to vector data
      call VecGetArray(solution,localptr,idx,ierr)

!  Simply write the solution directly into the array locations
!  Alternatively, we could use VecSetValues() or VecSetValuesLocal().

      do 10, i=mybase,myend
!       x - current location in global grid
        x = h*(i-1)
        localptr(i-mybase+idx+1) = (t + one_d0)*(one_d0 + x*x)
 10   continue

!  Restore vector

      call VecRestoreArray(solution,localptr,idx,ierr)

      return
      end

!  ------------------------------------------------------------------------
!
!   MyMonitor - A user-provided routine to monitor the solution computed at
!   each time-step.  This example plots the solution and computes the
!   error in two different norms.
!
!   Input Parameters:
!   ts     - the time-step context
!   step   - the count of the current step (with 0 meaning the
!            initial condition)
!   time   - the current time
!   u      - the solution at this timestep
!   dummy  - optional user-provided context for this monitoring
!            routine (not used here)
!
      subroutine MyMonitor(ts,step,time,u,dummy,ierr)
      implicit none
#include "ex2f.h"

!  Input/output parameters:
      TS       ts
      PetscInt step
      PetscObject dummy
      PetscReal time
      Vec      u
      PetscErrorCode ierr

!  Local variables:
      PetscScalar en2,en2s,enmax
      PetscScalar mone
      PetscDraw   draw

      mone = -1.0d0

!  We use the default X windows viewer
!       PETSC_VIEWER_DRAW_WORLD
!  that is associated with the PETSC_COMM_WORLD communicator. This
!  saves the effort of calling PetscViewerDrawOpen() to create the window.
!  Note that if we wished to plot several items in separate windows we
!  would create each viewer with PetscViewerDrawOpen() and store them in
!  the application context, appctx.
!
!  Double buffering makes graphics look better.

      call PetscViewerDrawGetDraw(PETSC_VIEWER_DRAW_WORLD,0,draw,ierr)
      call PetscDrawSetDoubleBuffer(draw,ierr)
      call VecView(u,PETSC_VIEWER_DRAW_WORLD,ierr)

!  Compute the exact solution at this time-step.  Note that the
!  solution vector is passed via the user-defined common block.
      call ExactSolution(time,ierr)

!  Print debugging information if desired
      if (debug) then
        if (rank .eq. 0) write(6,*) 'Computed solution vector'
        call VecView(u,PETSC_VIEWER_STDOUT_WORLD,ierr)
        if (rank .eq. 0) write(6,*) 'Exact solution vector'
        call VecView(solution,PETSC_VIEWER_STDOUT_WORLD,ierr)
      endif

!  Compute the 2-norm and max-norm of the error
      call VecAXPY(solution,mone,u,ierr)
      call VecNorm(solution,NORM_2,en2,ierr)

!  Scale the 2-norm by the grid spacing
      en2s = sqrt(dble(h))*en2
      call VecNorm(solution,NORM_MAX,enmax,ierr)

!  Print only from processor 0
      if (rank .eq. 0) write(6,100) step,time,en2s,enmax
 100  format('Timestep = ',i5,',time = ',f8.3,                          &
     &       ' sec, error [2-norm] = ',e10.3,                            &
     &       ', error [max-norm] = ',e10.3)

!  Print debugging information if desired
      if (debug) then
        if (rank .eq. 0) write(6,*) 'Error vector'
        call VecView(solution,PETSC_VIEWER_STDOUT_WORLD,ierr)
      endif

      return
      end

!  ------------------------------------------------------------------------
!
!   RHSFunction - User-provided routine that evalues the RHS function
!   in the ODE.  This routine is set in the main program by calling
!   TSSetRHSFunction().  We compute:
!         global_out = F(global_in)
!
!   Input Parameters:
!      ts         - timestep context
!      t          - current time
!      global_in  - input vector to function
!      dummy      - (optional) user-provided context for function evaluation
!                   (not used here because we use a common block instead)
!
!   Output Parameter:
!      global_out - value of function
!
      subroutine RHSFunction(ts,t,global_in,global_out,dummy)
      implicit none
#include "ex2f.h"

!  Input/output parameters:
      TS        ts
      PetscReal t
      Vec       global_in,global_out
      integer   dummy

!  Local variables:
      PetscScalar localptr(1),copyptr(1),sc
      PetscErrorCode ierr
      PetscInt    i,localsize
      PetscOffset idx_c,idx_l,il

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Get ready for local function computations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Scatter ghost points to local vector, using the 2-step process
!     DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
!  By placing code between these two statements, computations can be
!  done while messages are in transition.

      call DMGlobalToLocalBegin(da,global_in,INSERT_VALUES,u_local,ierr)
      call DMGlobalToLocalEnd(da,global_in,INSERT_VALUES,u_local,ierr)

!  Access directly the values in our local INPUT work vector
      call VecGetArray(u_local,localptr,idx_l,ierr)

!  Access directly the values in our local OUTPUT work vector
      call VecGetArray(localwork,copyptr,idx_c,ierr)

      sc = one_d0/(h*h*two_d0*(one_d0+t)*(one_d0+t))

!  Evaluate our function on the nodes owned by this processor
      call VecGetLocalSize(u_local,localsize,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Compute entries for the locally owned part
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Handle boundary conditions: This is done by using the boundary condition
!        u(t,boundary) = g(t,boundary)
!  for some function g. Now take the derivative with respect to t to obtain
!        u_{t}(t,boundary) = g_{t}(t,boundary)
!
!  In our case, u(t,0) = t + 1, so that u_{t}(t,0) = 1
!          and  u(t,1) = 2t+ 1, so that u_{t}(t,1) = 2

       if (rank .eq. 0)      copyptr(1+idx_c) = one_d0
       if (rank .eq. size-1) copyptr(localsize+idx_c) = two_d0

!  Handle the interior nodes where the PDE is replace by finite
!  difference operators.

      do 10, i=2,localsize-1
         il = i + idx_l
         copyptr(i+idx_c) =  localptr(il) * sc *                        &
     &     (localptr(il+1) + localptr(il-1) - two_d0*localptr(il))
 10   continue

      call VecRestoreArray(u_local,localptr,idx_l,ierr)
      call VecRestoreArray(localwork,copyptr,idx_c,ierr)

!  Return the values from our local OUTPUT vector into our global
!  output vector.

      call DMLocalToGlobalBegin(da,localwork,INSERT_VALUES,global_out   &
     &                          ,ierr)
      call DMLocalToGlobalEnd(da,localwork,INSERT_VALUES,global_out,    &
     &                          ierr)

!  Print debugging information if desired
      if (debug) then
        if (rank .eq. 0) write(6,*) 'RHS function vector'
        call VecView(global_out,PETSC_VIEWER_STDOUT_WORLD,ierr)
      endif

      return
      end

!  ------------------------------------------------------------------------
!
!  RHSJacobian - User-provided routine to compute the Jacobian of the
!                right-hand-side function.
!
!  Input Parameters:
!     ts - the TS context
!     t - current time
!     global_in - global input vector
!     dummy - optional user-defined context, as set by TSetRHSJacobian()
!
!  Output Parameters:
!     A - Jacobian matrix
!     B - optionally different preconditioning matrix
!     str - flag indicating matrix structure
!
!  Notes:
!  RHSJacobian computes entries for the locally owned part of the Jacobian.
!   - Currently, all PETSc parallel matrix formats are partitioned by
!     contiguous chunks of rows across the processors. The "grow"
!     parameter computed below specifies the global row number
!     corresponding to each local grid point.
!   - Each processor needs to insert only elements that it owns
!     locally (but any non-local elements will be sent to the
!     appropriate processor during matrix assembly).
!   - Always specify global row and columns of matrix entries.
!   - Here, we set all entries for a particular row at once.
!   - Note that MatSetValues() uses 0-based row and column numbers
!     in Fortran as well as in C.

      subroutine RHSJacobian(ts,t,global_in,A,B,str,dummy)
      implicit none
#include "ex2f.h"

!  Input/output parameters:
      TS               ts
      PetscReal t
      Vec              global_in
      Mat              A,B
      MatStructure     str
      integer          dummy

!  Local variables:
      PetscScalar localptr(1),sc,v(3)
      PetscInt    i,col(3),i1,i3
      PetscErrorCode ierr
      PetscInt    mstart(1),mend(1)
      PetscInt    mstarts,mends
      PetscOffset idx,is

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Get ready for local Jacobian computations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Scatter ghost points to local vector, using the 2-step process
!     DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
!  By placing code between these two statements, computations can be
!  done while messages are in transition.

      call DMGlobalToLocalBegin(da,global_in,INSERT_VALUES,u_local,ierr)
      call DMGlobalToLocalEnd(da,global_in,INSERT_VALUES,u_local,ierr)

!  Get pointer to vector data
      call VecGetArray(u_local,localptr,idx,ierr)

!  Get starting and ending locally owned rows of the matrix
      call MatGetOwnershipRange(A,mstarts,mends,ierr)
      mstart(1) = mstarts
      mend(1)   = mends

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Compute entries for the locally owned part of the Jacobian.
!   - Currently, all PETSc parallel matrix formats are partitioned by
!     contiguous chunks of rows across the processors.
!   - Each processor needs to insert only elements that it owns
!     locally (but any non-local elements will be sent to the
!     appropriate processor during matrix assembly).
!   - Here, we set all entries for a particular row at once.
!   - We can set matrix entries either using either
!     MatSetValuesLocal() or MatSetValues().
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Set matrix rows corresponding to boundary data
      i1 = 1
      i3 = 3
      if (mstart(1) .eq. 0) then
        v(1) = zero_d0
        call MatSetValues(A,i1,mstart,i1,mstart,v,INSERT_VALUES,ierr)
        mstart(1) = mstart(1) + 1
      endif
      if (mend(1) .eq. m) then
        mend(1) = mend(1) - 1
        v(1) = zero_d0
        call MatSetValues(A,i1,mend,i1,mend,v,INSERT_VALUES,ierr)
      endif

!  Set matrix rows corresponding to interior data.
!  We construct the matrix one row at a time.

      sc = one_d0/(h*h*two_d0*(one_d0+t)*(one_d0+t))
      do 10, i=mstart(1),mend(1)-1
         col(1) = i-1
         col(2) = i
         col(3) = i+1
         is     = i - mstart(1) + 1 +idx + 1
         v(1)   = sc*localptr(is)
         v(2)   = sc*(localptr(is+1) + localptr(is-1) -                 &
     &                four_d0*localptr(is))
         v(3)   = sc*localptr(is)
        call MatSetValues(A,i1,i,i3,col,v,INSERT_VALUES,ierr)
 10   continue

!  Restore vector
      call VecRestoreArray(u_local,localptr,idx,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Complete the matrix assembly process and set some options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Assemble matrix, using the 2-step process:
!       MatAssemblyBegin(), MatAssemblyEnd()
!  Computations can be done while messages are in transition,
!  by placing code between these two statements.

      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)

!  Set flag to indicate that the Jacobian matrix retains an identical
!  nonzero structure throughout all timestepping iterations (although the
!  values of the entries change). Thus, we can save some work in setting
!  up the preconditioner (e.g., no need to redo symbolic factorization for
!  ILU/ICC preconditioners).
!   - If the nonzero structure of the matrix is different during
!     successive linear solves, then the flag DIFFERENT_NONZERO_PATTERN
!     must be used instead.  If you are unsure whether the matrix
!     structure has changed or not, use the flag DIFFERENT_NONZERO_PATTERN.
!   - Caution:  If you specify SAME_NONZERO_PATTERN, PETSc
!     believes your assertion and does not check the structure
!     of the matrix.  If you erroneously claim that the structure
!     is the same when it actually is not, the new preconditioner
!     will not function correctly.  Thus, use this optimization
!     feature with caution!

      str = SAME_NONZERO_PATTERN

!  Set and option to indicate that we will never add a new nonzero location
!  to the matrix. If we do, it will generate an error.

      call MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE,ierr)

      return
      end
