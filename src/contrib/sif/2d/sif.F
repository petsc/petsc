!
! "$Id: sif.F,v 1.2 1998/11/05 17:20:39 balay Exp $";
!/*T
!   Concepts: TS^time-dependent nonlinear problems
!   Routines: TSCreate(); TSSetSolution(); TSSetRHSFunction(); TSSetRHSJacobian();
!   Routines: TSSetType(); TSSetInitialTimeStep(); TSSetDuration();
!   Routines: TSSetFromOptions(); TSStep(); TSDestroy(); TSMonitorSet();
!   Routines: PetscPrintf();
!   Processors: n
!T*/
!
!  ------------------------------------------------------------------------
!
!   This program solves a time-dependent nonlinear PDE using implicit
!   timestepping (the method of lines).  This code uses Sundials to do the
!   time integration.
!
!   The plan is for this code to support both 1-dimensional and 
!   2-dimensional versions.  Currently, only the 1-dimensional version
!   is fully supported, but some infrastructure for the 2-dimensional
!   variant is in place.  Use the option -dim2 to activate the 2D version.
!
!   Additional modifications needed for 2D:
!     - Fix code to use the M,N,MN parameters correctly
!     - Add new routines for function/initial guess evaluation
!
!  --------------------------------------------------------------------------

      program main
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!                    Include files
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!
!  Each routine within this program uses the include file 'sif.h',
!  which itself includes the various PETSc include files as well as
!  problem-specific data in several common blocks.
!
!  This program uses CPP for preprocessing, as indicated by the use of
!  PETSc include files in the directory petsc/include/FINCLUDE.  This
!  convention enables use of the CPP preprocessor, which allows the use
!  of the #include statements that define PETSc objects and variables.
!
!  Use of the conventional Fortran include statements is also supported
!  In this case, the PETsc include files are located in the directory
!  petsc/include/finclude.
!         
!  Since one must be very careful to include each file no more than once
!  in a Fortran routine, application programmers must explicitly list
!  each file needed for the various PETSc components within their
!  program (unlike the C/C++ interface).
!
!  See the Fortran section of the PETSc users manual for details.

#include "sif.h"

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!
!  Variables:
!     ts         - timestepping solver
!     A          - Jacobian matrix context
!     local      - local vector
!     global     - global vector
!     ftime      - final time
!     duration   - total duration
!     time_steps - number of timesteps
!
!     coloring and fd are used by PETSc to efficiently compute the sparse 
!     Jacobian of the application function.
!
!  Note that vectors are declared as PETSc "Vec" objects.  These vectors
!  are mathematical objects that contain more than just an array of
!  double precision numbers. I.e., vectors in PETSc are not just
!        double precision x(*).
!  However, local vector data can be easily accessed via VecGetArray().
!  See the Fortran section of the PETSc users manual for details.

      TS                ts
      Vec               local, global
      Mat               A
      ISColoring        coloring
      MatFDColoring     fd

      integer           flg, ierr
      integer           time_steps, steps
      double precision  dt, ftime, duration
      
      double precision  tda


!  Note: Any user-defined Fortran routines (such as RHSFunction)
!  MUST be declared as external.

      external Monitor, RHSFunction
      external InitialConditions

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      comm = PETSC_COMM_WORLD
      call MPI_Comm_size(comm,size,ierr)
      call MPI_Comm_rank(comm,rank,ierr)

!  Initialize problem parameters
!
!  use -ts_max_steps and -ts_max_time to override limits
!
      time_steps = 1000
      duration   = 1.0e-4
!
      M          = 60
      N          = 1
      debug      = 0
      dimension  = 1

      call OptionsGetInt(PETSC_NULL_CHARACTER,'-M',M,flg,ierr)
      call OptionsHasName(PETSC_NULL_CHARACTER,'-debug',debug,ierr)

!     Flag to activate 2-dimensional problem
      call OptionsHasName(PETSC_NULL_CHARACTER,'-dim2',flg,ierr)  
      if (flg .eq. 1) then
         dimension = 2
         call OptionsGetInt(PETSC_NULL_CHARACTER,'-N',N,flg,ierr)
         write(6,*) 'ERROR: This code is NOT finished for the 2D case!'
         goto 999
      endif

!     Compute total number of grid points
      MN = M * N

!     Set the parameters in the U_t common block

        hi1  = real(M - 2)
        hi2  = hi1*hi1
        h    = 1.0d0/hi1
        d1p1 = .5d0*hi1
        d2p2 = .5d0*hi2

        str     = 0.0d0
        dep     = 0.0d0
        tda     = 3.0d0
        rpar(2) = - tan(1.57079632679490d0 - tda/2.0d0)
        rpar(4) = 1.0d0/sqrt(1.0d0 + rpar(2)**2)
        rpar(5) = 1.d0
        rpar(6) = .05d0

!
!    This timestep is not used by Sundials but the simple Backward Euler scheme
!  does need an initial time-step
!
        dt      = .5d0*h

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create vector data structures
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create distributed array (DA) to manage parallel grid and vectors
!  Set up the ghost point communication pattern.  There are M total
!  grid values in the first dimension spread equally among all the 
!  processors; likewise, there are N grid values in the second dimension.
!  Then get local grid boundaries.

      if (dimension .eq. 1) then
         call DACreate1d(MPI_COMM_WORLD,DA_NONPERIODIC,M,1,2,           &
     &        PETSC_NULL_INTEGER,da,ierr)
         call DAGetCorners(da,xs,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER, &
     &        xm,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr)
         call DAGetGhostCorners(da,gxs,PETSC_NULL_INTEGER,              &
     &        PETSC_NULL_INTEGER,gxm,PETSC_NULL_INTEGER,                &
     &        PETSC_NULL_INTEGER,ierr)
         
      else if (dimension .eq. 2) then
         call DACreate2d(MPI_COMM_WORLD,DA_NONPERIODIC,                 &
     &        DA_STENCIL_BOX,M,N,                                       &
     &        PETSC_DECIDE,PETSC_DECIDE,1,2,                            &
     &        PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,da,ierr)
         call DAGetCorners(da,xs,ys,PETSC_NULL_INTEGER,xm,ym,           &
     &        PETSC_NULL_INTEGER,ierr)
         call DAGetGhostCorners(da,gxs,gys,PETSC_NULL_INTEGER,gxm,gym,  &
     &        PETSC_NULL_INTEGER,ierr)
      else
         write(6,*) 'ERROR: Only dimensions 1 and 2 supported.'
         goto 999
      endif

!  Here we shift the starting indices up by one so that we can easily
!  use the Fortran convention of 1-based indices (rather 0-based indices).

      xs  = xs+1
      gxs = gxs+1
      xe  = xs+xm-1
      gxe = gxs+gxm-1

      if (dimension .eq. 1) then
         ys  = -1
         ym  = -1
         ye  = -1
         gys = -1
         gym = -1
         gye = -1
      else
         ys  = ys+1
         gys = gys+1
         ye  = ys+ym-1
         gye = gys+gym-1
      endif

!  
!  Extract global and local vectors from DA; then duplicate for remaining
!  vectors that are the same types.

      call DAGetDistributedVector(da,global,ierr)
      call DAGetLocalVector(da,local,ierr)


!  Make global work vector for storing exact solution
      call VecDuplicate(global,solution,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Evaluate initial conditions
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call InitialConditions(global)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create timestepping solver context; set various callback routines
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call TSCreate(comm,TS_NONLINEAR,ts,ierr)
      call TSMonitorSet(ts,Monitor,PETSC_NULL,ierr)
      call TSSetRHSFunction(ts,RHSFunction,PETSC_NULL,ierr)
      call TSSetType(ts,TS_SUNDIALS,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create matrix data structure; set Jacobian evaluation routine;
!  This uses the PETSc built in routines for computing sparse 
!  Jacobians efficiently.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call DAGetColoring(da,coloring,A,ierr)
      call MatFDColoringCreate(A,coloring,fd,ierr)
      call ISColoringDestroy(coloring,ierr)
      call MatFDColoringSetFunction(fd,RHSFunction,PETSC_NULL,ierr)

      call TSSetRHSJacobianDefault(ts,fd,A,A,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Customize timestepping solver; set runtime options.
!  We can override the defaults set by TSSetDuration() with
!     -ts_max_steps <maxsteps> -ts_max_time <maxtime>
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call TSSetDuration(ts,time_steps,duration,ierr)
      call TSSetFromOptions(ts,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Set solution vector, initial timestep.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call TSSetInitialTimeStep(ts,0.0d0,dt,ierr)
      call TSSetSolution(ts,global,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Set up and run the timestepping solver
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call TSSetUp(ts,ierr)
      call TSStep(ts,steps,ftime,ierr)

!
!    Save the solution to a file called solution in a format that
!    Matlab can easily read in
!
      call ViewerFileOpenASCII(comm,'solution',output,ierr)
      call ViewerSetFormat(output,VIEWER_FORMAT_ASCII_MATLAB,           &
     &                     'solution',ierr)
      call VecView(global,output,ierr)
      call ViewerDestroy(output,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call TSDestroy(ts,ierr)
      call VecDestroy(solution,ierr)
      call VecDestroy(local,ierr)
      call VecDestroy(global,ierr)
      call MatFDColoringDestroy(fd,ierr)
      call DADestroy(da,ierr)
      call MatDestroy(A,ierr)

!  Always call PetscFinalize() before exiting a program.  This routine
!    - finalizes the PETSc libraries as well as MPI
!    - provides summary and diagnostic information if certain runtime
!      options are chosen (e.g., -log_summary).

 999  continue
      call PetscFinalize(ierr)
      end

!  ------------------------------------------------------------------------
!
!  InitialConditions - Computes the solution at the initial time. 
!
!  Input Parameter:
!     global - uninitialized solution vector
!
!  Output Parameter:
!     global - vector with solution at initial time
!
      subroutine InitialConditions(global)
      implicit none
#include "sif.h"

!  Input/output parameters:
      Vec     global

!  Local variables:
      double precision y(1), x
      integer          i, mybase, myend, ierr, idx
      double precision r1,sqt,cp,rc,orig,one,p1

!  Determine starting and ending point of each processor's range of
!  grid values.  Note that we shift by 1 to convert from the 0-based
!  C convention of starting indices to the 1-based Fortran convention.

      call VecGetOwnershipRange(global,mybase,myend,ierr)
      mybase = mybase + 1

      h = 1.0d0/(M-2)

!  Get a pointer to vector data.
!    - For default PETSc vectors, VecGetArray() returns a pointer to
!      the data array.  Otherwise, the routine is implementation dependent.
!    - You MUST call VecRestoreArray() when you no longer need access to
!      the array.
!    - Note that the Fortran interface to VecGetArray() differs from the
!      C version.  See the users manual for details.

      call VecGetArray(global,y,idx,ierr) 

!     We initialize the solution array by simply writing the solution
!     directly into the array locations.

       one  = 1.0d0
       r1   = 1.d0
       sqt  = sqrt(2.d0)
       cp   = .05d0
       
       rc       = - r1/sqt
       orig     = rc + sqrt(one - cp*cp)*r1
       p1       = r1*r1
       if (mybase .eq. 1) then 
         write (8,*) 'New origin at y=0 x=',rc
         write (8,*) 'Radius =',r1
         write (8,*) 'Centered at'
         write (8,*) 'x =',rc,'  y = 0'
         write (8,*) 'Initial neck/radius =',cp
       endif
!
!    Last value in vector gets funny non-pysical value
!
       if (myend .eq. M) then
         y(myend+1-mybase+idx) = orig
         myend                 = myend - 1
       endif
!
!      loop over vector elements stored on this processor
!    from mybase to myend and put in the initial condition
!
       do 20 i=mybase,myend
         x                 = real(i-1)*h
         y(i+1-mybase+idx) = sqrt(p1 - (x*orig - rc)**2)
 20    continue 


!  Restore vector

      call VecRestoreArray(global,y,idx,ierr) 

      call VecView(global,VIEWER_DRAWX_WORLD,ierr)

      return
      end

!  ------------------------------------------------------------------------
!
!   Monitor - A user-provided routine to monitor the solution computed at 
!   each time-step.  This example plots the solution and computes the
!   error in two different norms.
!
!   Input Parameters:
!   ts     - the time-step context
!   step   - the count of the current step (with 0 meaning the
!            initial condition)
!   time   - the current time
!   global - vector containing the current solution
!   dummy  - optional user-provided context for this monitoring routine,
!           not used here
!
!   Output Parameter:
!   global - the solution at this timestep

      subroutine Monitor(ts,step,time,global,dummy)
      implicit none
#include "sif.h"

!  Input/output parameters:
      TS                ts
      integer           step, dummy
      double precision  time
      Vec               global

!  Local variables:
      integer  ierr
      Draw     draw


!  We use the default X windows viewer
!       VIEWER_DRAWX_WORLD
!  that is associated with the PETSC_COMM_WORLD communicator. This
!  saves the effort of calling ViewerDrawOpenX() to create the window.
!  Note that if we wished to plot several items in separate windows we
!  would create each viewer with ViewerDrawOpenX() and store them in
!  the application context, appctx.
!
!  Double buffering makes graphics look better.

      call ViewerDrawGetDraw(VIEWER_DRAWX_WORLD,draw,ierr)
      call DrawSetDoubleBuffer(draw,ierr)
      call VecView(global,VIEWER_DRAWX_WORLD,ierr)


!  Print debugging information if desired
      if (debug .eq. 1) then
        call ViewerPushFormat(VIEWER_STDOUT_WORLD,                      &
     &                   VIEWER_FORMAT_ASCII_MATLAB,'solution',ierr)
        call VecView(global,VIEWER_STDOUT_WORLD,ierr)
        call ViewerPopFormat(VIEWER_STDOUT_WORLD,ierr)
      endif

!  Print only from processor 0
      if (rank .eq. 0) then 
        write(6,100) step,time
      endif
 100  format('Timestep = ',i5,', time = ',e9.3,' sec')

      return
      end

!  ------------------------------------------------------------------------
!
!   RHSFunction - User-provided routine that evalues the RHS function
!   in the ODE.  This routine is set in the main program by calling
!   TSSetRHSFunction().  We compute:
!         globalout = F(globalin)
!
!   Input Parameters:
!      ts        - timestep context
!      t         - current time
!      globalin  - input vector to function
!      dummy     - (optional) user-provided context for function
!                  (not used here because we use a common block instead)
!
!   Output Parameter:
!      globalout - value of function
!
      subroutine RHSFunction(ts,t,globalin,globalout,dummy,ierr)
      implicit none
#include "sif.h"

!  Input/output parameters:
      TS               ts
      double precision t
      Vec              globalin, globalout
      integer          dummy

!  Local variables:
      Vec              local
      double precision localptr(1), copyptr(1)
      integer          ierr, idx_c, idx_l, localsize

!  Print debugging information if desired

      if (debug .eq. 1) then
        call ViewerPushFormat(VIEWER_STDOUT_WORLD,                      &
     &        VIEWER_FORMAT_ASCII_MATLAB,'globalin',ierr)
        call VecView(globalin,VIEWER_STDOUT_WORLD,ierr)
        call ViewerPopFormat(VIEWER_STDOUT_WORLD,ierr)
      endif

!  The vector 'local' will be a workspace that contains the ghost region

      call DAGetLocalVector(da,local,ierr)
  
!  Copy the input vector into local and up-date the ghost points
      call DAGlobalToLocalBegin(da,globalin,INSERT_VALUES,local,ierr)
      call DAGlobalToLocalEnd(da,globalin,INSERT_VALUES,local,ierr)

!  Access directly the values in our local INPUT work array
      call VecGetArray(local,localptr,idx_l,ierr) 

!  Access directly the values in our local OUTPUT work array
      call VecGetArray(globalout,copyptr,idx_c,ierr) 

!  Evaluate our function on the nodes owned by this processor

      call VecGetLocalSize(local,localsize,ierr)
      call res2(localsize,t,localptr(idx_l+1),copyptr(idx_c+1))

      call VecRestoreArray(local,localptr,idx_l,ierr) 
      call VecRestoreArray(globalout,copyptr,idx_c,ierr) 

!  Print debugging information if desired

      if (debug .eq. 1) then
        call ViewerPushFormat(VIEWER_STDOUT_WORLD,                      &
     &        VIEWER_FORMAT_ASCII_MATLAB,'globalout',ierr)
        call VecView(globalout,VIEWER_STDOUT_WORLD,ierr)
        call ViewerPopFormat(VIEWER_STDOUT_WORLD,ierr)
      endif

      return
      end

! **********************************************************************
!
! Parallel version
!
	subroutine res2 (neq,t,y,del)
        implicit double precision (a-h,o-z)
#include "sif.h"
!
!  Evaluate the time derivatives resulted in spatial discretization of
!  the equation for surface movement due to diffusion in surface and  
!  grain boundary.
!  Smearing discretization scheme
!  Fixed dihedral angle DA, m = rpar(2) = ctg(DA/2)
!  Cartisian coordinates.
!  Close to a quarter of a circle
!

        double precision t, y(*),del(*)
        integer          neq

        double precision bcastvalues(3),yneq
        integer          ierr,istart,iend

!
!    yneq - contains the value of y at the last mesh point (which all processors need)
!
        double precision half,one,two,three,four,five,six,ten,sqt
        double precision zero,op5

        parameter (half=0.5d0,one=1.d0,two=2.d0,three=3.d0,four=4.d0)
        parameter (five=5.d0,six=6.d0,ten=10.d0)
        parameter (sqt=1.41421356237310d0)                                      
        parameter (zero=0.d0,op5=1.5d0)
        
!
!      Note that y is of length gxm since it includes the ghost points while
!           del is of length xm since it DOES NOT include the ghost points
!
        if (rank .eq. size-1) then
!
! For the processor handling the BCs at the junction.   
!
!     y(gxm) is the last entry in the global vector
!            corresponds to y(neq) in the sequential code
!
               d0s  = y(gxm)*y(gxm) 
! Fictitious point from fixed DA
               d1   = rpar(2)*y(gxm)
               yp1  = y(gxm-2) + d1/d1p1
! Curvatures at the tip
               d2   = (yp1 - two*y(gxm-1) + y(gxm-2))*hi2
               d1s  = d1*d1
               cur  = one/(d0s + d1s)
               cp   = - d2*cur**op5
               dp   = cur**half
               d1o  = d1
! O*Flux/B at the tip (from grain boundary by continuity condition)
               yab   = y(gxm-1)
               fltip = dep*(rpar(4) - cp*y(gxm)*yab + str)/yab**2.d0
! Time derivative of the shifting. 
               del(xm) = - fltip/yab 
               par1    = del(xm)*h/y(gxm)
!
!
! Calculate the time derivative at the adjacent mesh point.
               d1  = (y(gxm-1) - y(gxm-3))*d1p1
               d2  = (y(gxm-1) - two*y(gxm-2) + y(gxm-3))*hi2
               d1s = d1*d1
               cur = one/(d0s + d1s)
               ci  = - d2*cur**op5
               di  = cur**half
               d1p = d1
!
               d1  = (y(gxm-2) - y(gxm-4))*d1p1
               d2  = (y(gxm-2) - two*y(gxm-3) + y(gxm-4))*hi2
               d1s = d1*d1
               cur = one/(d0s + d1s)
               cm  = - d2*cur**op5
               dm  = cur**half
! f0 = flux/y(gxm)
               f0  = fltip/y(gxm)
               f1h = (dm + di)*(ci - cm)*d1p1

!
!    The IBM compiler with -O3 optimization compiles 
!   f0h = (dp + di)*(cp - ci)*d1p1 incorrectly. 
!
               f0h = -(dp + di)*(ci - cp)*d1p1

               f1  = (f0h + f1h)*half
!
               del(xm-1) = hi1*(f1 - four*f0h + three*f0)
               del(xm-1) = del(xm-1) + d1o*par1*(60.d0 - 2.0d0)      
               del(xm-2) = d2p2*((cp - ci)*(dp + di) -                  &
     &             (ci - cm)*(di + dm)) + par1*d1p*(60.d0 - 3.0d0) 

        endif

! Broadcast par1 and d0s to all processors, from the final processor (size - 1)

        bcastvalues(1) = par1
        bcastvalues(2) = d0s
        bcastvalues(3) = y(gxm)
        call MPI_Bcast(bcastvalues,3,MPI_DOUBLE_PRECISION, size-1,comm, &
     &                 ierr)
        par1 = bcastvalues(1)
        d0s  = bcastvalues(2)
        yneq = bcastvalues(3)

!
! For all processors
! 
      istart = xs
      iend   = xe
      if (rank .eq. 0) then 
        istart = 3
      endif
      if (rank .eq. size-1) then
        iend   = xe-3
      endif

!
!   the i iterates over the global indices of y for this processor
!
      do 10 i=istart,iend
!
        d1  = (y(i-gxs+1) - y(i-gxs+1-2))*d1p1
        d2  = (y(i-gxs+1) - two*y(i-gxs+1-1) + y(i-gxs+1-2))*hi2
        d1s = d1*d1
        cur = one/(d0s + d1s)
        cm  = - d2*cur**op5
        dm  = cur**half
!
        d1  = (y(i-gxs+1+1) - y(i-gxs+1-1))*d1p1
        d2  = (y(i-gxs+1+1) - two*y(i-gxs+1) + y(i-gxs+1-1))*hi2
        d1s = d1*d1
        cur = one/(d0s + d1s)
        ci  = - d2*cur**op5
        di  = cur**half
	d1i = d1
!
        d1  = (y(i-gxs+1+2) - y(i-gxs+1))*d1p1
        d2  = (y(i-gxs+1) - two*y(i-gxs+1+1) + y(i-gxs+1+2))*hi2
        d1s = d1*d1
        cur = one/(d0s + d1s)
        cp  = - d2*cur**op5
        dp  = cur**half
!
	del(i-xs+1) = d2p2*((cp - ci)*(dp + di) - (ci - cm)*(di + dm))    &
     &       + par1*d1i*(i-1.0d0)
 10   continue
!

        if (rank .eq. 0) then
!
! For the processor handling the end of symmetry. 
!
! Fictitious points from symmetry
           p1  = - yneq*h
           ym1 = sqrt(rpar(5) - (p1 - rpar(6))**2) 
           p1  =  p1*two  
           ym2 = sqrt(rpar(5) - (p1 - rpar(6))**2) 
!
           d1  = (y(1) - ym2)*d1p1
           d2  = (ym2 - two*ym1 + y(1))*hi2
           d1s = d1*d1
           cur = one/(d0s + d1s)
           cm  = - d2*cur**op5
           dm  = cur**half
!
           d1  = (y(2) - ym1)*d1p1
           d2  = (ym1 - two*y(1) + y(2))*hi2
           d1s = d1*d1
           cur = one/(d0s + d1s)
           ci  = - d2*cur**op5
           di  = cur**half
!
           d1  = (y(3) - y(1))*d1p1
           d2  = (y(3) - two*y(2) + y(1))*hi2
           d1s = d1*d1
           cur = one/(d0s + d1s)
           cp  = - d2*cur**op5
           dp  = cur**half
           d1i = d1
!
           del(1) = d2p2*((cp - ci)*(dp + di) - (ci - cm)*(di + dm))
!
           d1  = (y(4) - y(2))*d1p1
           d2  = (y(4) - two*y(3) + y(2))*hi2
           d1s = d1*d1
           cur = one/(d0s + d1s)
           cf  = - d2*cur**op5
           df  = cur**half
!
           del(2) = d2p2*((cf - cp)*(dp + df) - (cp - ci)*(di + dp))     &
     &               + par1*d1i
!
        endif

!
        return
	end


