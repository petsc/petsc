C
C "$Id: ex2f.F,v 1.13 1998/03/01 03:56:54 curfman Exp bsmith $";
C/*T
C   Concepts: TS^time-dependent nonlinear problems
C   Routines: TSCreate(); TSSetSolution(); TSSetRHSFunction(); TSSetRHSJacobian();
C   Routines: TSSetType(); TSSetInitialTimeStep(); TSSetDuration();
C   Routines: TSSetFromOptions(); TSStep(); TSDestroy(); TSSetMonitor();
C   Routines: PetscPrintf();
C   Processors: n
CT*/
C
C  ------------------------------------------------------------------------
C
C   This program solves a simple time-dependent nonlinear PDE using implicit
C   timestepping:
C                                    U * U_xx 
C                              U_t = ---------
C                                    2*(t+1)^2 
C
C             U(0,x) = 1 + x*x; U(t,0) = t + 1; U(t,1) = 2*t + 2
C
C   The exact solution is U(t,x) = (1 + x*x) * (1 + t).
C
C   Note that since the solution is linear in time and quadratic in x, 
C   the finite difference scheme actually computes the "exact" solution.
C
C   We use the backward Euler method.
C
C  --------------------------------------------------------------------------

      program main
      implicit none

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C                    Include files
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C
C  Each routine within this program uses the include file 'ex2f.h',
C  which itself includes the various PETSc include files as well as
C  problem-specific data in several common blocks.
C
C  This program uses CPP for preprocessing, as indicated by the use of
C  PETSc include files in the directory petsc/include/FINCLUDE.  This
C  convention enables use of the CPP preprocessor, which allows the use
C  of the #include statements that define PETSc objects and variables.
C
C  Use of the conventional Fortran include statements is also supported
C  In this case, the PETsc include files are located in the directory
C  petsc/include/finclude.
C         
C  Since one must be very careful to include each file no more than once
C  in a Fortran routine, application programmers must exlicitly list
C  each file needed for the various PETSc components within their
C  program (unlike the C/C++ interface).
C
C  See the Fortran section of the PETSc users manual for details.

#include "ex2f.h"

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C                   Variable declarations
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C
C  Variables:
C     ts         - timestepping solver
C     A          - Jacobian matrix context
C     local      - local vector
C     global     - global vector
C     ftime      - final time
C     duration   - total duration
C     time_steps - number of timesteps
C
C  Note that vectors are declared as PETSc "Vec" objects.  These vectors
C  are mathematical objects that contain more than just an array of
C  double precision numbers. I.e., vectors in PETSc are not just
C        double precision x(*).
C  However, local vector data can be easily accessed via VecGetArray().
C  See the Fortran section of the PETSc users manual for details.

      TS               ts
      Vec              local, global
      Mat              A
      integer          flg, ierr
      integer          time_steps, steps
      double precision dt, ftime, duration

C  Note: Any user-defined Fortran routines (such as RHSFunction)
C  MUST be declared as external.

      external Monitor, RHSFunction
      external InitialConditions, RHSJacobian, RHSJacobianFD
      external RHSJacobianFD

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C                 Beginning of program
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      comm = PETSC_COMM_WORLD
      call MPI_Comm_size(comm,size,ierr)
      call MPI_Comm_rank(comm,rank,ierr)

C  Initialize problem parameters

      time_steps = 1000
      duration   = 100.0
      M          = 60
      debug      = 0
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-M',M,flg,ierr)
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-time',time_steps,
     &                   flg,ierr)
      call OptionsHasName(PETSC_NULL_CHARACTER,'-debug',debug,ierr)
      one_d0  = 1.0d0
      two_d0  = 2.0d0
      four_d0 = 4.0d0
      h       = one_d0/(M-one_d0)
      dt      = h/two_d0

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Create vector data structures
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

C  Create distributed array (DA) to manage parallel grid and vectors
C  Set up the ghost point communication pattern.  There are M total
C  grid values spread equally among all the processors.

      call DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,M,1,1,
     &     PETSC_NULL,da,ierr)

C  Extract global and local vectors from DA; then duplicate for remaining
C  vectors that are the same types.

      call DAGetDistributedVector(da,global,ierr)
      call DAGetLocalVector(da,local,ierr)

C  Make local work vector for evaluating right-hand-side function
      call VecDuplicate(local,localwork,ierr)

C  Make global work vector for storing exact solution
      call VecDuplicate(global,solution,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Evaluate initial conditions
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      if (rank .eq. 0) 
     &  write(6,*) 'Calling InitialConditions() routine ...'
      call InitialConditions(global)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Create timestepping solver context; set various callback routines
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      if (rank .eq. 0) write(6,*) 'Creating TS solver ...'
      call TSCreate(comm,TS_NONLINEAR,ts,ierr)
      call TSSetMonitor(ts,Monitor,PETSC_NULL,ierr)
      call TSSetRHSFunction(ts,RHSFunction,PETSC_NULL,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Create matrix data structure; set Jacobian evaluation routine;
C  Set the solution method to be backward Euler.
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call MatCreate(comm,M,M,A,ierr)
      call OptionsHasName(PETSC_NULL_CHARACTER,'-fdjac',flg,ierr)
      if (flg .eq. 1) then
         call SetCRoutineFromFortran(ts,A,A,ierr)
      else
         call TSSetRHSJacobian(ts,A,A,RHSJacobian,PETSC_NULL,ierr)
      endif
      call TSSetType(ts,TS_BEULER,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Customize timestepping solver; set runtime options.
C  We can override the defaults set by TSSetDuration() with
C     -ts_max_steps <maxsteps> -ts_max_time <maxtime>
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call TSSetDuration(ts,time_steps,duration,ierr)
      call TSSetFromOptions(ts,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Set solution vector, initial timestep.
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call TSSetInitialTimeStep(ts,0.0,dt,ierr)
      call TSSetSolution(ts,global,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Set up and run the timestepping solver
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      if (rank .eq. 0) write(6,*) 'Calling timestepping solver ...'
      call TSSetUp(ts,ierr)
      call TSStep(ts,steps,ftime,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Free work space.  All PETSc objects should be destroyed when they
C  are no longer needed.
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      if (rank .eq. 0) write(6,*) 'Freeing work space ...'
      call TSDestroy(ts,ierr)
      call VecDestroy(localwork,ierr)
      call VecDestroy(solution,ierr)
      call VecDestroy(local,ierr)
      call VecDestroy(global,ierr)
      call DADestroy(da,ierr)
      call MatDestroy(A,ierr)

C  Always call PetscFinalize() before exiting a program.  This routine
C    - finalizes the PETSc libraries as well as MPI
C    - provides summary and diagnostic information if certain runtime
C      options are chosen (e.g., -log_summary).

      call PetscFinalize(ierr)
      end

C  ------------------------------------------------------------------------
C
C  InitialConditions - Computes the solution at the initial time. 
C
C  Input Parameter:
C     global - uninitialized solution vector
C
C  Output Parameter:
C     global - vector with solution at initial time
C
      subroutine InitialConditions(global)
      implicit none
#include "ex2f.h"

C  Input/output parameters:
      Vec     global

C  Local variables:
      double  precision localptr(1), x
      integer           i, mybase, myend, ierr
      PetscOffset       idx

C  Determine starting and ending point of each processor's range of
C  grid values.  Note that we shift by 1 to convert from the 0-based
C  C convention of starting indices to the 1-based Fortran convention.

      call VecGetOwnershipRange(global,mybase,myend,ierr)
      mybase = mybase + 1

C  Get a pointer to vector data.
C    - For default PETSc vectors, VecGetArray() returns a pointer to
C      the data array.  Otherwise, the routine is implementation dependent.
C    - You MUST call VecRestoreArray() when you no longer need access to
C      the array.
C    - Note that the Fortran interface to VecGetArray() differs from the
C      C version.  See the users manual for details.

      call VecGetArray(global,localptr,idx,ierr) 

C     We initialize the solution array by simply writing the solution
C     directly into the array locations.

      do 10, i=mybase,myend
C       x - current location in global grid 
        x = h*(i-1)
        localptr(i-mybase+idx+1) = one_d0 + x*x
 10   continue

C  Restore vector

      call VecRestoreArray(global,localptr,idx,ierr) 

C  Print debugging information if desired
      if (debug .eq. 1) then
        if (rank .eq. 0) write(6,*) 'initial guess vector'
        call VecView(global,VIEWER_STDOUT_WORLD,ierr)
      endif

      return
      end

C  ------------------------------------------------------------------------
C
C  ExactSolution - Computes the exact solution at any time.
C
C  Input Parameter:
C    t - time
C
C  Output Parameter:
C    ierr - error code
C
C  Note: The solution vector is stored in the common block!
C
      subroutine ExactSolution(t,ierr)
      implicit none
#include "ex2f.h"

C  Input/output parameters:
      double precision  t, x
      integer           ierr

C  Local variables:
      double precision  localptr(1)
      integer           i, mybase, myend
      PetscOffset       idx

C  Determine starting and ending point of each processors range of
C  grid values.  Note that we shift by 1 to convert from the 0-based
C  C convention of starting indices to the 1-based Fortran convention.

      call VecGetOwnershipRange(solution,mybase,myend,ierr)
      mybase = mybase + 1

C  Get a pointer to vector data
      call VecGetArray(solution,localptr,idx,ierr) 

C  Simply write the solution directly into the array locations

      do 10, i=mybase,myend
C       x - current location in global grid 
        x = h*(i-1)
        localptr(i-mybase+idx+1) = (t + one_d0)*(one_d0 + x*x)
 10   continue

C  Restore vector

      call VecRestoreArray(solution,localptr,idx,ierr) 

      return
      end

C  ------------------------------------------------------------------------
C
C   Monitor - A user-provided routine to monitor the solution computed at 
C   each time-step.  This example plots the solution and computes the
C   error in two different norms.
C
C   Input Parameters:
C   ts     - the time-step context
C   step   - the count of the current step (with 0 meaning the
C            initial condition)
C   time   - the current time
C   dummy  - optional user-provided context for this monitoring routine,
C           not used here
C
C   Output Parameter:
C   global - the solution at this timestep

      subroutine Monitor(ts,step,time,global,dummy)
      implicit none
#include "ex2f.h"

C  Input/output parameters:
      TS               ts
      integer          step, dummy
      double precision time
      Vec              global

C  Local variables:
      integer          ierr
      double precision en2, en2s, enmax
      double precision mone
      Draw             draw

      mone = -1.0d0

C  We use the default X windows viewer
C       VIEWER_DRAWX_PETSC_COMM 
C  that is associated with the PETSC_COMM_WORLD communicator. This
C  saves the effort of calling ViewerDrawOpenX() to create the window.
C  Note that if we wished to plot several items in separate windows we
C  would create each viewer with ViewerDrawOpenX() and store them in
C  the application context, appctx.
C
C  Double buffering makes graphics look better.

      call ViewerDrawGetDraw(VIEWER_DRAWX_WORLD,draw,ierr)
      call DrawSetDoubleBuffer(draw,ierr)
      call VecView(global,VIEWER_DRAWX_WORLD,ierr)

C  Compute the exact solution at this time-step, then compute the 
C  2-norm and max-norm of the error.  Note that the solution vector
C  is passed via the user-defined common block.
      call ExactSolution(time,ierr)

C  Print debugging information if desired
      if (debug .eq. 1) then
        if (rank .eq. 0) write(6,*) 'Computed solution vector'
        call VecView(global,VIEWER_STDOUT_WORLD,ierr)
        if (rank .eq. 0) write(6,*) 'Exact solution vector'
        call VecView(solution,VIEWER_STDOUT_WORLD,ierr)
      endif

C  Compute the 2-norm and max-norm of the error
      call VecAXPY(mone,global,solution,ierr)
      call VecNorm(solution,NORM_2,en2,ierr)
C  Scale the 2-norm by the grid spacing
      en2s = dsqrt(h)*en2
      call VecNorm(solution,NORM_MAX,enmax,ierr)

C  Print only from processor 0
      if (rank .eq. 0) write(6,100) step,time,en2s,enmax
 100  format('Timestep = ',i5,', time = ',f8.3,
     &       ' sec, error [2-norm] = ',e9.3,
     &       ', error [max-norm] = ',e9.3)

C  Print debugging information if desired
      if (debug .eq. 1) then
        if (rank .eq. 0) write(6,*) 'Error vector'
        call VecView(solution,VIEWER_STDOUT_WORLD,ierr)
      endif

      return
      end

C  ------------------------------------------------------------------------
C
C   RHSFunction - User-provided routine that evalues the RHS function
C   in the ODE.  This routine is set in the main program by calling
C   TSSetRHSFunction().  We compute:
C         globalout = F(globalin)
C
C   Input Parameters:
C      ts        - timestep context
C      t         - current time
C      globalin  - input vector to function
C      dummy     - (optional) user-provided context for function
C                  (not used here because we use a common block instead)
C
C   Output Parameter:
C      globalout - value of function
C
      subroutine RHSFunction(ts,t,globalin,globalout,dummy)
      implicit none
#include "ex2f.h"

C  Input/output parameters:
      TS               ts
      double precision t
      Vec              globalin, globalout
      integer          dummy

C  Local variables:
      Vec              local
      double precision localptr(1), copyptr(1), sc
      integer          i, il, ierr, localsize
      PetscOffset      idx_c, idx_l

C  The vector 'local' will be a workspace that contains the ghost region

      call DAGetLocalVector(da,local,ierr)
  
C  Copy the input vector into local and up-date the ghost points
      call DAGlobalToLocalBegin(da,globalin,INSERT_VALUES,local,ierr)
      call DAGlobalToLocalEnd(da,globalin,INSERT_VALUES,local,ierr)

C  Access directly the values in our local INPUT work array
      call VecGetArray(local,localptr,idx_l,ierr) 

C  Access directly the values in our local OUTPUT work array
      call VecGetArray(localwork,copyptr,idx_c,ierr) 

      sc = one_d0/(h*h*two_d0*(one_d0+t)*(one_d0+t))

C  Evaluate our function on the nodes owned by this processor

      call VecGetLocalSize(local,localsize,ierr)

C  Handle boundary conditions: This is done by using the boundary condition 
C        U(t,boundary) = g(t,boundary) 
C  for some function g. Now take the derivative with respect to t to obtain
C        U_{t}(t,boundary) = g_{t}(t,boundary)
C
C  In our case, U(t,0) = t + 1; so U_{t}(t,0) = 1 
C          and  U(t,1) = 2t+ 1; so U_{t}(t,1) = 2

       if (rank .eq. 0)      copyptr(1+idx_c) = one_d0
       if (rank .eq. size-1) copyptr(localsize+idx_c) = two_d0

C  Handle the interior nodes where the PDE is replace by finite 
C  difference operators.

      do 10, i=2, localsize-1
         il = i + idx_l
         copyptr(i+idx_c) =  localptr(il) * sc * 
     &     (localptr(il+1) + localptr(il-1) - two_d0*localptr(il))
 10   continue

      call VecRestoreArray(local,localptr,idx_l,ierr) 
      call VecRestoreArray(localwork,copyptr,idx_c,ierr) 

C  Return the values from our local OUTPUT array into our global 
C  output array.

      call DALocalToGlobal(da,localwork,INSERT_VALUES,globalout,ierr)

C  Print debugging information if desired
      if (debug .eq. 1) then
        if (rank .eq. 0) write(6,*) 'RHS function vector'
        call VecView(globalout,VIEWER_STDOUT_WORLD,ierr)
      endif

      return
      end

C  ------------------------------------------------------------------------
C
C  RHSJacobian - User-provided routine to compute the Jacobian of the
C                right-hand-side function.
C
C  Input Parameters:
C     ts - the TS context
C     t - current time
C     globalin - global input vector
C     dummy - optional user-defined context, as set by TSetRHSJacobian()
C
C  Output Parameters:
C     A - Jacobian matrix
C     B - optionally different preconditioning matrix
C     str - flag indicating matrix structure
C
C  Notes:
C  RHSJacobian computes entries for the locally owned part of the Jacobian.
C   - Currently, all PETSc parallel matrix formats are partitioned by
C     contiguous chunks of rows across the processors. The "grow"
C     parameter computed below specifies the global row number 
C     corresponding to each local grid point.
C   - Each processor needs to insert only elements that it owns
C     locally (but any non-local elements will be sent to the
C     appropriate processor during matrix assembly). 
C   - Always specify global row and columns of matrix entries.
C   - Here, we set all entries for a particular row at once.
C   - Note that MatSetValues() uses 0-based row and column numbers
C     in Fortran as well as in C.

      subroutine RHSJacobian(ts,t,globalin,A,B,str,dummy)
      implicit none
#include "ex2f.h"

C  Input/output parameters:
      TS               ts
      double precision t
      Vec              globalin
      Mat              A, B
      MatStructure     str
      integer          dummy

C  Local variables:
      Vec              local
      double precision localptr(1), sc, v(3)
      integer          i, ierr,  col(3), is
      integer          mstart, mend, mstarts, mends
      PetscOffset      idx

C  Extract local array
      call DAGetLocalVector(da,local,ierr)
      call DAGlobalToLocalBegin(da,globalin,INSERT_VALUES,local,ierr)
      call DAGlobalToLocalEnd(da,globalin,INSERT_VALUES,local,ierr)

C  Get a pointer to vector data

      call VecGetArray(local,localptr,idx,ierr) 

C  Set matrix sparsity info

      str = SAME_NONZERO_PATTERN

C  Get starting and ending locally owned rows of the matrix

      call MatGetOwnershipRange(A,mstarts,mends,ierr)
      mstart = mstarts
      mend   = mends

C  Set matrix rows corresponding to boundary data

      if (mstart .eq. 0) then
        v(1) = zero_d0
        call MatSetValues(A,1,mstart,1,mstart,v,INSERT_VALUES,ierr)
        mstart = mstart + 1
      endif
      if (mend .eq. M) then
        mend = mend - 1
        v(1) = zero_d0
        call MatSetValues(A,1,mend,1,mend,v,INSERT_VALUES,ierr)
      endif

C  Set matrix rows corresponding to interior data.
C  We construct the matrix one row at a time.

      sc = one_d0/(h*h*two_d0*(one_d0+t)*(one_d0+t))
      do 10, i=mstart, mend-1
         col(1) = i-1
         col(2) = i
         col(3) = i+1
         is     = i - mstart + 1 +idx + 1
         v(1)   = sc*localptr(is)
         v(2)   = sc*(localptr(is+1) + localptr(is-1) -
     &                four_d0*localptr(is))
         v(3)   = sc*localptr(is)
        call MatSetValues(A,1,i,3,col,v,INSERT_VALUES,ierr)
 10   continue

C  Restore vector

      call VecRestoreArray(local,localptr,idx,ierr) 

C  Assemble matrix, using the 2-step process:
C       MatAssemblyBegin(), MatAssemblyEnd()
C  Computations can be done while messages are in transition,
C  by placing code between these two statements.

      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)

      return
      end

