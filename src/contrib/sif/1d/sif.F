C
C "$Id: sif.F,v 1.9 1999/11/02 13:04:53 curfman Exp $";
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
C   This program solves a time-dependent nonlinear PDE using implicit
C   timestepping (the method of lines):
C
C   Uses Sundials to do the time integration 
C
C  --------------------------------------------------------------------------

      program main

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C                    Include files
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C
C  Each routine within this program uses the include file 'sif.h',
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
C  in a Fortran routine, application programmers must explicitly list
C  each file needed for the various PETSc components within their
C  program (unlike the C/C++ interface).
C
C  See the Fortran section of the PETSc users manual for details.

#include "sif.h"

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
C     coloring and fd are used by PETSc to efficiently compute the sparse 
C     Jacobian of the application function.
C
C  Note that vectors are declared as PETSc "Vec" objects.  These vectors
C  are mathematical objects that contain more than just an array of
C  double precision numbers. I.e., vectors in PETSc are not just
C        double precision x(*).
C  However, local vector data can be easily accessed via VecGetArray().
C  See the Fortran section of the PETSc users manual for details.

      TS                ts
      Vec               local, global
      Mat               A
      ISColoring        coloring
      MatFDColoring     fd
      Draw              draw

      integer           flg, ierr
      integer           time_steps, steps, kdim
      double precision  dt, ftime, duration, errabs, errrel, faclin

C  Note: Any user-defined Fortran routines (such as RHSFunction)
C  MUST be declared as external.

      external Monitor, RHSFunction
      external InitialConditions

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C                 Beginning of program
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      comm = PETSC_COMM_WORLD
      call MPI_Comm_size(comm,size,ierr)
      call MPI_Comm_rank(comm,rank,ierr)

C  Initialize problem parameters
C
C  use -ts_max_steps and -ts_max_time to override limits
C
      time_steps = 100000
      duration   = 1
C
      M          = 130
C      M          = 60
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-M',M,flg,ierr)
      call OptionsHasName(PETSC_NULL_CHARACTER,'-debug',debug,ierr)
      errabs     = 1.d-10
      errrel     = 1.d-10
      faclin     = 0.0005d0
      kdim       = M  
      debug      = 0

C  Double buffering makes graphics look better.
C
      call ViewerDrawGetDraw(VIEWER_DRAW_WORLD,0,draw,ierr)
      call DrawSetDoubleBuffer(draw,ierr)

C
C     Set the parameters in the U_t common block
C
        zero_d0   = 0.d0
        one_d0    = 1.d0
        two_d0    = 2.d0
        three_d0  = 3.d0
        four_d0   = 4.d0
        hi1  = M - 2
        hi2  = hi1*hi1
        h    = one_d0/hi1
        d1p1 = hi1/two_d0
        d2p2 = hi2/two_d0
C
        rpar(2) = zero_d0
C
C    This timestep is not used by Sundials but the simple Backward Euler scheme
C  does need an initial time-step
C
        dt      = h/two_d0

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Create vector data structures
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

C  Create distributed array (DA) to manage parallel grid and vectors
C  Set up the ghost point communication pattern.  There are M total
C  grid values spread equally among all the processors.

      call DACreate1d(MPI_COMM_WORLD,DA_NONPERIODIC,M,1,2,
     &     PETSC_NULL_INTEGER,da,ierr)

C  Get local grid boundaries (for 1-dimensional DA)

      call DAGetCorners(da,xs,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER
     &                  ,xm,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER
     &                  ,ierr)
      call DAGetGhostCorners(da,gxs,PETSC_NULL_INTEGER
     &                       ,PETSC_NULL_INTEGER,gxm,
     &                       PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,ierr)

C  Here we shift the starting indices up by one so that we can easily
C  use the Fortran convention of 1-based indices (rather 0-based indices).

      xs  = xs+1
      gxs = gxs+1
      xe  = xs+xm-1
      gxe = gxs+gxm-1

C  
C  Extract global and local vectors from DA; then duplicate for remaining
C  vectors that are the same types.

      call DACreateGlobalVector(da,global,ierr)
      call DACreateLocalVector(da,local,ierr)


C  Make global work vector for storing exact solution
      call VecDuplicate(global,solution,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Evaluate initial conditions
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call InitialConditions(global)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Create timestepping solver context; set various callback routines
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call TSCreate(comm,TS_NONLINEAR,ts,ierr)
      call TSSetMonitor(ts,Monitor,PETSC_NULL_DOUBLE,ierr)
      call TSSetRHSFunction(ts,RHSFunction,PETSC_NULL_DOUBLE,ierr)
      call TSSetType(ts,TS_SUNDIALS,ierr)
      call TSSundialsSetTolerance(ts,errabs,errrel,ierr)
      call TSSundialsSetLinearTolerance(ts,faclin,ierr)
      call TSSundialsSetGMRESRestart(ts,kdim,ierr)
      if (rank .eq. 0)  then
      write (6,*) 'Use PETSc'
      write (6,*) 'errabs, errrel = ', errabs, errrel
      write (6,*) '# of mesh points', M-1
      write (6,*) 'K_dimension',kdim
      write (6,*) 'Factor of linear tolerance',faclin  
      endif

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Create matrix data structure; set Jacobian evaluation routine;
C  This uses the PETSc built in routines for computing sparse 
C  Jacobians efficiently.
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call DAGetColoring(da,coloring,A,ierr)
      call MatFDColoringCreate(A,coloring,fd,ierr)
      call ISColoringDestroy(coloring,ierr)
      call MatFDColoringSetFunctionTS(fd,RHSFunction
     &         ,PETSC_NULL_DOUBLE,ierr)

      call TSSetRHSJacobian(ts,A,A,TSDefaultComputeJacobianColor,
     &         fd,ierr)

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

      call TSSetInitialTimeStep(ts,0.0d0,dt,ierr)
      call TSSetSolution(ts,global,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Set up and run the timestepping solver
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call TSSetUp(ts,ierr)
      call TSStep(ts,steps,ftime,ierr)

C
C    Save the solution to a file called solution in a format that
C    Matlab can easily read in
C
      call ViewerASCIIOpen(comm,'solution.m',output,ierr)
      call ViewerSetFormat(output,VIEWER_FORMAT_ASCII_MATLAB,
     &                     'solution',ierr)
      call VecView(global,output,ierr)
      call ViewerDestroy(output,ierr)

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  Free work space.  All PETSc objects should be destroyed when they
C  are no longer needed.
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call TSDestroy(ts,ierr)
      call VecDestroy(solution,ierr)
      call VecDestroy(local,ierr)
      call VecDestroy(global,ierr)
      call MatFDColoringDestroy(fd,ierr)
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
#include "sif.h"

C  Input/output parameters:
      Vec     global

C  Local variables:
      double precision y(1), x
      integer          i, mybase, myend, ierr
      PetscOffset      idx
      double precision twopi 

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

      call VecGetArray(global,y,idx,ierr) 

C     We initialize the solution array by simply writing the solution
C     directly into the array locations.

       if (mybase .eq. 1) then 
         write (6,*) 'Initial surface is a sinusoid curve'
       endif
C
C    Last value in vector gets funny non-pysical value
C
       twopi = 4*asin(one_d0)
       if (myend .eq. M) then
         y(myend+1-mybase+idx) = twopi         
         myend                 = myend - 1
       endif
C
C      loop over vector elements stored on this processor
C    from mybase to myend and put in the initial condition
C
       do 20 i=mybase,myend
         x                 = (i-1)*h
         y(i+1-mybase+idx) = cos(x*twopi)
 20    continue 


C  Restore vector

      call VecRestoreArray(global,y,idx,ierr) 

      call VecView(global,VIEWER_DRAW_WORLD,ierr)

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
C   global - vector containing the current solution
C   dummy  - optional user-provided context for this monitoring routine,
C           not used here
C
C   Output Parameter:
C   global - the solution at this timestep

      subroutine Monitor(ts,step,time,global,dummy)
#include "sif.h"

C  Input/output parameters:
      TS                ts
      integer           step, dummy,nonlin,lin
      double precision  time
      Vec               global

C  Local variables:
      integer  ierr


C  We use the default X windows viewer
C       VIEWER_DRAW_WORLD
C  that is associated with the PETSC_COMM_WORLD communicator. This
C  saves the effort of calling ViewerDrawOpenX() to create the window.
C  Note that if we wished to plot several items in separate windows we
C  would create each viewer with ViewerDrawOpenX() and store them in
C  the application context, appctx.
C
      call VecView(global,VIEWER_DRAW_WORLD,ierr)


C  Print debugging information if desired
      if (debug .eq. 1) then
        call ViewerPushFormat(VIEWER_STDOUT_WORLD,
     &                   VIEWER_FORMAT_ASCII_MATLAB,'solution',ierr)
        call VecView(global,VIEWER_STDOUT_WORLD,ierr)
        call ViewerPopFormat(VIEWER_STDOUT_WORLD,ierr)
      endif

      call TSSundialsGetIterations(ts,nonlin,lin,ierr)
C  Print only from processor 0
      if (rank .eq. 0) then 
        write(6,100) step,time,nonlin,lin
      endif
 100  format('Timestep = ',i6,', time = ',e16.8,' Newton Steps = '
     &       ,i6,' linear its = ',i6)

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
      subroutine RHSFunction(ts,t,globalin,globalout,dummy,ierr)
#include "sif.h"

C  Input/output parameters:
      TS               ts
      double precision t
      Vec              globalin, globalout
      integer          dummy

C  Local variables:
      Vec              local
      double precision localptr(1), copyptr(1)
      integer          ierr, localsize
      PetscOffset      idx_l,idx_c

C  Print debugging information if desired

      if (debug .eq. 1) then
        call ViewerPushFormat(VIEWER_STDOUT_WORLD,
     &                   VIEWER_FORMAT_ASCII_MATLAB,'globalin',ierr)
        call VecView(globalin,VIEWER_STDOUT_WORLD,ierr)
        call ViewerPopFormat(VIEWER_STDOUT_WORLD,ierr)
      endif

C  The vector 'local' will be a workspace that contains the ghost region

      call DACreateLocalVector(da,local,ierr)
  
C  Copy the input vector into local and up-date the ghost points
      call DAGlobalToLocalBegin(da,globalin,INSERT_VALUES,local,ierr)
      call DAGlobalToLocalEnd(da,globalin,INSERT_VALUES,local,ierr)

      call VecGetLocalSize(local,localsize,ierr)

C  Access directly the values in our local INPUT work array
      call VecGetArray(local,localptr,idx_l,ierr) 

C  Access directly the values in our local OUTPUT work array
      call VecGetArray(globalout,copyptr,idx_c,ierr) 

C  Evaluate our function on the nodes owned by this processor

      call res2(localsize,t,localptr(idx_l+1),copyptr(idx_c+1))

      call VecRestoreArray(local,localptr,idx_l,ierr) 
      call VecRestoreArray(globalout,copyptr,idx_c,ierr) 

C  Print debugging information if desired

      if (debug .eq. 1) then
        call ViewerPushFormat(VIEWER_STDOUT_WORLD,
     &                   VIEWER_FORMAT_ASCII_MATLAB,'globalout',ierr)
        call VecView(globalout,VIEWER_STDOUT_WORLD,ierr)
        call ViewerPopFormat(VIEWER_STDOUT_WORLD,ierr)
      endif

      call VecDestroy(local,ierr)
      return
      end

c **********************************************************************
c
c Parallel version
c
	subroutine res2 (neq,t,y,del)
#include "sif.h"
c
c  Evaluate the time derivatives resulted in spatial discretization of
c  the equation for surface movement due to diffusion in surface and  
c  grain boundary.
c
        double precision t,y(*),del(*)
        double precision half,one,two,three,four
        double precision zero,op5
        parameter (half=0.5d0,one=1.d0,two=2.d0,three=3.d0,four=4.d0)
        parameter (zero=0.d0,op5=1.5d0)
c
        double precision d0s,d1,d2,yp1,dp,cur,d1o,yab,fltip,par1,d1p,f1h
     &                   ,dm,di,cm,ci,f1,f0h,f0,p1,ym1,ym2,d1s,cp
     &                   ,cf,df,d1i
c
        integer          i,neq
c
        double precision bcastvalues(3),yneq
        integer          ierr,istart,iend
c
c    yneq - contains the value of y at the last mesh point (which all processors need)
c
c      Note that y is of length gxm since it includes the ghost points while
c           del is of length xm since it DOES NOT include the ghost points
c
        if (rank .eq. size-1) then
c
c For the processor handling the BCs at the junction.   
c
c     y(gxm) is the last entry in the global vector
c            corresponds to y(neq) in the sequential code
c
               d0s  = y(gxm)*y(gxm) 
c Fictitious point from fixed DA
               d1   = rpar(2)*y(gxm)
               yp1  = y(gxm-2) + d1/d1p1
c Curvatures at the tip
               d2   = (yp1 - two*y(gxm-1) + y(gxm-2))*hi2
               d1s  = d1*d1
               cur  = one/(d0s + d1s)
               cp   = - d2*cur**op5
               dp   = cur**half
               d1o  = d1
c O*Flux/B at the tip (from grain boundary by continuity condition)
               yab   = y(gxm-1)
               fltip = zero 
c Time derivative of the shifting. 
               del(xm) = - fltip/yab 
               par1    = del(xm)*h/y(gxm)
c
c Calculate the time derivative at the adjacent mesh point.
               d1  = (y(gxm-1) - y(gxm-3))*d1p1
               d2  = (y(gxm-1) - two*y(gxm-2) + y(gxm-3))*hi2
               d1s = d1*d1
               cur = one/(d0s + d1s)
               ci  = - d2*cur**op5
               di  = cur**half
               d1p = d1
c
               d1  = (y(gxm-2) - y(gxm-4))*d1p1
               d2  = (y(gxm-2) - two*y(gxm-3) + y(gxm-4))*hi2
               d1s = d1*d1
               cur = one/(d0s + d1s)
               cm  = - d2*cur**op5
               dm  = cur**half
c f0 = flux/y(gxm)
               f0  = fltip/y(gxm)
               f1h = (dm + di)*(ci - cm)*d1p1

C
C    The IBM compiler with -O3 optimization compiles 
C   f0h = (dp + di)*(cp - ci)*d1p1 incorrectly. 
C
               f0h = -(dp + di)*(ci - cp)*d1p1

               f1  = (f0h + f1h)*half
c
               del(xm-1) = hi1*(f1 - four*f0h + three*f0)
c               del(xm-1) = del(xm-1) + d1o*par1*(60.d0 - 2.0d0)      
               del(xm-1) = del(xm-1) + d1o*par1*(M*1.0d0 - 2.0d0)      
               del(xm-2) = d2p2*((cp - ci)*(dp + di) - 
c     &             (ci - cm)*(di + dm)) + par1*d1p*(60.d0 - 3.0d0) 
     &             (ci - cm)*(di + dm)) + par1*d1p*(M*1.0d0 - 3.0d0) 

        endif

c Broadcast par1 and d0s to all processors, from the final processor (size - 1)

        bcastvalues(1) = par1
        bcastvalues(2) = d0s
        bcastvalues(3) = y(gxm)
        call MPI_Bcast(bcastvalues,3,MPI_DOUBLE_PRECISION, size-1,comm,
     &                 ierr)
        par1 = bcastvalues(1)
        d0s  = bcastvalues(2)
        yneq = bcastvalues(3)

c
c For all processors
c 
      istart = xs
      iend   = xe
      if (rank .eq. 0) then 
        istart = 3
      endif
      if (rank .eq. size-1) then
        iend   = xe-3
      endif

C
c   the i iterates over the global indices of y for this processor
c
      do 10 i=istart,iend
c
        d1  = (y(i-gxs+1) - y(i-gxs+1-2))*d1p1
        d2  = (y(i-gxs+1) - two*y(i-gxs+1-1) + y(i-gxs+1-2))*hi2
        d1s = d1*d1
        cur = one/(d0s + d1s)
        cm  = - d2*cur**op5
        dm  = cur**half
c
        d1  = (y(i-gxs+1+1) - y(i-gxs+1-1))*d1p1
        d2  = (y(i-gxs+1+1) - two*y(i-gxs+1) + y(i-gxs+1-1))*hi2
        d1s = d1*d1
        cur = one/(d0s + d1s)
        ci  = - d2*cur**op5
        di  = cur**half
	d1i = d1
c
        d1  = (y(i-gxs+1+2) - y(i-gxs+1))*d1p1
        d2  = (y(i-gxs+1) - two*y(i-gxs+1+1) + y(i-gxs+1+2))*hi2
        d1s = d1*d1
        cur = one/(d0s + d1s)
        cp  = - d2*cur**op5
        dp  = cur**half
c
	del(i-xs+1) = d2p2*((cp - ci)*(dp + di) - (ci - cm)*(di + dm))
     &                + par1*d1i*(i-1.0d0)
 10   continue
c

        if (rank .eq. 0) then
c
c For the processor handling the end of symmetry. 
c
c Fictitious points from symmetry
           p1  = - yneq*h
           ym1 = y(2)                              
           p1  =  p1*two  
           ym2 = y(3)                              
c
           d1  = (y(1) - ym2)*d1p1
           d2  = (ym2 - two*ym1 + y(1))*hi2
           d1s = d1*d1
           cur = one/(d0s + d1s)
           cm  = - d2*cur**op5
           dm  = cur**half
c
           d1  = (y(2) - ym1)*d1p1
           d2  = (ym1 - two*y(1) + y(2))*hi2
           d1s = d1*d1
           cur = one/(d0s + d1s)
           ci  = - d2*cur**op5
           di  = cur**half
c
           d1  = (y(3) - y(1))*d1p1
           d2  = (y(3) - two*y(2) + y(1))*hi2
           d1s = d1*d1
           cur = one/(d0s + d1s)
           cp  = - d2*cur**op5
           dp  = cur**half
           d1i = d1
c
           del(1) = d2p2*((cp - ci)*(dp + di) - (ci - cm)*(di + dm))
c
           d1  = (y(4) - y(2))*d1p1
           d2  = (y(4) - two*y(3) + y(2))*hi2
           d1s = d1*d1
           cur = one/(d0s + d1s)
           cf  = - d2*cur**op5
           df  = cur**half
c
           del(2) = d2p2*((cf - cp)*(dp + df) - (cp - ci)*(di + dp))
     &               + par1*d1i
c
        endif

c
        return
	end
