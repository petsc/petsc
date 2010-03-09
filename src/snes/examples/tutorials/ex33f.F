> 
      program radhyd
! "$Id: ex4f.F,v 1.39 1999/03/10 19:29:25 Vince Mousseau $";
!
!  Description: This example solves a nonlinear system on 1 processor with SNES.
!  We solve the Euler equations in one dimension.
!  The command line options include:
!    -dt <dt>, where <dt> indicates time step
!    -mx <xg>, where <xg> = number of grid points in the x-direction
!    -nstep <nstep>, where <nstep> = number of time steps
!    -debug <ndb>, where <ndb> = 0) no debug 1) debug
!    -pcnew <npc>, where <npc> = 0) no preconditioning 1) rad preconditioning
!    -probnum <probnum>, where <probnum> = 1) cyclic Riesner 2) dam break
!    -ihod <ihod>, where <ihod> = 1) upwind 2) quick 3) godunov
!    -ientro <ientro>, where <ientro> = 0) basic 1) entropy fix 2) hlle
!    -theta <theta>, where <theta> = 0-1 0-explicit 1-implicit
!    -hnaught <hnaught>, where <hnaught> = height of left side
!    -hlo <hlo>, where <hlo> = hieght of right side
!    -ngraph <ngraph>, where <ngraph> = number of time steps between graphics
!    -damfac <damfac>, where <damfac> = fractional downward change in hight
!    -dampit <ndamp>, where <ndamp> = 1 turn damping on
!    -gorder <gorder>, where <gorder> = spatial oerder of godunov
!
!
! 
--------------------------------------------------------------------------
! 
! Shock tube example
!
!  In this example the application context is a Fortran integer array:
!      ctx(1) = shell preconditioner pressure matrix contex
!      ctx(2) = semi implicit pressure matrix
!      ctx(4) = xold  - old time values need for time advancement
!      ctx(5) = mx    - number of control volumes
!      ctx(6) = N     - total number of unknowns
!
! 
--------------------------------------------------------------------------

      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!                    Include files
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!
!  The following include statements are generally used in SNES Fortran
!  programs:
!     petscsys.h  - base PETSc routines
!     vec.h    - vectors
!     mat.h    - matrices
!     ksp.h    - Krylov subspace methods
!     pc.h     - preconditioners
!     snes.h   - SNES interface
!  In addition, we need the following for use of PETSc drawing routines
!     draw.h   - drawing routines
!  Other include statements may be needed if using additional PETSc
!  routines in a Fortran program, e.g.,
!     viewer.h - viewers
!     is.h     - index sets
!
#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscis.h"
#include "finclude/petscdraw.h"
#include "finclude/petscmat.h"
#include "finclude/petscksp.h"
#include "finclude/petscpc.h"
#include "finclude/petscsnes.h"
#include "finclude/petscviewer.h"

#include "comd.h"
#include "tube.h"

!
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!
!  Variables:
!     snes        - nonlinear solver
!     x,r         - solution, residual vectors
!     J           - Jacobian matrix
!     its         - iterations for convergence
!     matrix_free - flag - 1 indicates matrix-free version
!     dt          - time step size
!     draw        - drawing context
!
      PetscFortranAddr   ctx(6)
      integer            nx,ny
      SNES               snes
      KSP                ksp
      PC                 pc
      Vec                x,r
      PetscViewer        view0,view1,view2,
     1                   view3, view4
      Mat                Psemi
      integer            matrix_free, flg, N, ierr, ngraph
      integer            nstep, ndt, size, rank, i
      integer            its, lits, totits, totlits
      integer            ndb, npc, ndamp, nwilson, ndtcon
      double precision   plotim
!      logical            pcnew

      double precision krtol,katol,kdtol
      double precision natol,nrtol,nstol
      integer  kmit,nmit,nmf


!  Note: Any user-defined Fortran routines (such as FormJacobian)
!  MUST be declared as external.

      external FormFunction, FormInitialGuess,FormDt, 
     &         PCRadSetUp, PCRadApply, FormGraph,FormDampit
      double precision eos

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!  Initialize program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      open (unit=87,file='Dt.out',status='unknown')

c
c  start PETSc
c
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)

      if (size .ne. 1) then
         if (rank .eq. 0) then
            write(6,*) 'This is a uniprocessor example only!'
         endif
         SETERRQ(1,' ',ierr)
      endif

!  Initialize problem parameters

      debug       = .false.
      dampit      = .false.
      wilson      = .true.
      dtcon       = .true.
      pcnew       = .true.
      dtmax       = 1.0d+2
      dtmin       = 1.00d-12
      dt          = 1.0d-2
      mx          = 100
      nstep       = 50
      matrix_free = 1
      probnum     = 1
      gorder      = 1

      tfinal      = 1.0d+0
      tplot       = 0.2d+0
      dtgrow      = 1.05d+0
      tcscal      = 0.5d+0
      hcscal      = 0.5d+0

      ihod = 3
      ientro = 1
      theta = 1.00d+0
      pi = 3.14159d+0

      zero = 0.0
      ngraph = 10

      ndb = 0
      npc = 1

      damfac = 0.9d+0

      gamma = 1.25d+0
      csubv = 1.0d+0 / (gamma - 1.0d+0)

      v1 = 0.0d+0
      v4 = 0.0d+0

      e1 = 1.0d+0
      e4 = 1.0d+0

      r1 = 1.0d+0
      r4 = 2.0d+0

      ru1 = r1 * v1
      ru4 = r4 * v4

      et1 = r1 * ( (0.5d+0 * v1 * v1) + e1 )
      et4 = r4 * ( (0.5d+0 * v4 * v4) + e4 )

      p1 = eos(r1,ru1,et1)
      p4 = eos(r4,ru4,et4)

      a1 = sqrt(gamma*p1/r1)
      a4 = sqrt(gamma*p4/r4)

      erg0   = 1.0d+2
      kappa0 = 1.0d+0
      kappaa = -2.0d+0
      kappab = 13.0d+0 / 2.0d+0
CVAM  kappab = 2.5d+0

c
c  load the command line options
c
      call PetscOptionsGetReal(PETSC_NULL_CHARACTER,'-dt',dt,flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-mx',mx,flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-nstep',nstep,flg,
     &                        ierr)
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-debug',ndb,flg,
     &                        ierr)
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-pcnew',npc,flg,
     &                        ierr)
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-ihod',ihod,flg,
     &                        ierr)
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-ientro',ientro,flg,
     &                        ierr)
      call PetscOptionsGetReal(PETSC_NULL_CHARACTER,'-theta',
     &                                            theta,flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-ngraph',ngraph,flg,
     &                        ierr)
      call PetscOptionsGetReal(PETSC_NULL_CHARACTER,
     &                            '-damfac',damfac,flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-dampit',ndamp,flg,
     &                        ierr)
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,
     &                                   '-wilson',nwilson,flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-gorder',gorder,flg,
     &                        ierr)
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,
     &                            '-probnum',probnum,flg,ierr)
      call PetscOptionsGetReal(PETSC_NULL_CHARACTER,
     &                            '-kappa0',kappa0,flg,ierr)
      call PetscOptionsGetReal(PETSC_NULL_CHARACTER,
     &                            '-erg0',erg0,flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-dtcon',ndtcon,flg,
     &                        ierr)
      call PetscOptionsGetReal(PETSC_NULL_CHARACTER,
     &                            '-tfinal',tfinal,flg,ierr)
      call PetscOptionsGetReal(PETSC_NULL_CHARACTER,
     &                            '-tplot',tplot,flg,ierr)
      call PetscOptionsGetReal(PETSC_NULL_CHARACTER,
     &                            '-dtgrow',dtgrow,flg,ierr)
      call PetscOptionsGetReal(PETSC_NULL_CHARACTER,
     &                            '-tcscal',tcscal,flg,ierr)
      call PetscOptionsGetReal(PETSC_NULL_CHARACTER,
     &                            '-hcscal',hcscal,flg,ierr)
      call PetscOptionsGetReal(PETSC_NULL_CHARACTER,
     &                            '-dtmax',dtmax,flg,ierr)

      if (ndamp .eq. 1) then
         dampit = .true.
      endif

      if (nwilson .eq. 0) then
         wilson = .false.
      endif

      if (ndb .eq. 1) then
         debug = .true.
      endif

      if (npc .eq. 0) then
         pcnew = .false.
      endif

      if (ndtcon .eq. 0) then
         dtcon = .false.
      endif

CVAM  if (dt .ge. dtmax .or. dt .le. dtmin) then
CVAM     if (rank .eq. 0) write(6,*) 'DT is out of range'
CVAM     SETERRA(1,0,' ')
CVAM  endif

      N       = mx*neq

      ctx(5) = mx
      ctx(6) = N

      if (debug) then
        write(*,*) 'mx = ',mx
      endif



! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create nonlinear solver context
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call SNESCreate(PETSC_COMM_WORLD,snes,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Create vector data structures; set function evaluation routine
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      CALL MatCreate(PETSC_COMM_WORLD, ctx(2), ierr)
      CALL MatSetSizes(ctx(2),PETSC_DECIDE,PETSC_DECIDE,mx,mx,ierr)
      CALL MatSetFromOptions(ctx(2),ierr)
      if (debug) then
        call MatGetSize(ctx(2),nx,ny,ierr)
        write(*,*) 'number of rows = ',nx,' number of col = ',ny
      endif
c
c  full size vectors
c
      CALL VecCreate(PETSC_COMM_WORLD,x,ierr)
      CALL VecSetSizes(x,PETSC_DECIDE,N,ierr)
      CALL VecSetType(x,VECMPI,ierr)
      call VecSetFromOptions(x,ierr)
      call VecDuplicate(x,r,ierr)
      call VecDuplicate(x,ctx(4),ierr)
c
c set grid
c
      dx = 2.0d+0/dfloat(mx)
      xl0 = -1.0d+0 -(0.5d+0 * dx)

      if (debug) then
        write(*,*) 'dx = ',dx
      endif
 

!  Set function evaluation routine and vector.  Whenever the nonlinear
!  solver needs to evaluate the nonlinear function, it will call this
!  routine.
!   - Note that the final routine argument is the user-defined
!     context that provides application-specific data for the
!     function evaluation routine.

      call SNESSetFunction(snes,r,FormFunction,ctx,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Customize nonlinear solver; set runtime options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Set runtime options (e.g., -snes_monitor -snes_rtol <rtol> -ksp_type 
<type>)

      call SNESSetFromOptions(snes,ierr)
c
c  set the line search function to damp the newton update.
c
!     if (dampit) then
!       call SNESSetLineSearch(snes,FormDampit,ctx,ierr)
!     endif
c
c get the linear solver info from the nonlinear solver
c

      call SNESGetKSP(snes,ksp,ierr)
      call KSPGetPC(ksp,pc,ierr)
!      call KSPGetKSP(ksp,ksp1,ierr)
CVAM  call KSPSetType(ksp,KSPPREONLY,ierr)
      call KSPSetType(ksp,KSPGMRES,ierr)

      call KSPGetTolerances(ksp,krtol,katol,kdtol,kmit,ierr)
      call SNESGetTolerances(snes,natol,nrtol,nstol,nmit,nmf,ierr)

      write(*,*) 
      write(*,*) 
      write(*,*) 'Linear solver'
      write(*,*) 
      write(*,*) 'rtol = ',krtol
      write(*,*) 'atol = ',katol
      write(*,*) 'dtol = ',kdtol
      write(*,*) 'maxits = ',kmit
      write(*,*) 
      write(*,*) 
      write(*,*) 'Nonlinear solver' 
      write(*,*) 
      write(*,*) 'rtol = ',nrtol
      write(*,*) 'atol = ',natol
      write(*,*) 'stol = ',nstol
      write(*,*) 'maxits = ',nmit
      write(*,*) 'max func = ',nmf
      write(*,*) 
      write(*,*)

c
c  Build shell based preconditioner if flag set
c
      if (pcnew) then
        call PCSetType(pc,PCSHELL,ierr)
        call PCShellSetContext(pc,ctx,ierr)
        call PCShellSetSetUp(pc,PCRadSetUp,ierr)
        call PCShellSetApply(pc,PCRadApply,ierr)
      endif

      call PCCreate(PETSC_COMM_WORLD,ctx(1),ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Evaluate initial guess; then solve nonlinear system.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
c
c  initial counters
c
      time = 0.0d+0
      plotim = 0.0d+0
      totits = 0
      totlits = 0

!  Note: The user should initialize the vector, x, with the initial guess
!  for the nonlinear solver prior to calling SNESSolve().  In particular,
!  to employ an initial guess of zero, the user should explicitly set
!  this vector to zero by calling VecSet().

      call FormInitialGuess(x,ierr)
c
c  open a window to plot results
c
      call PetscViewerDrawOpen(PETSC_COMM_WORLD,PETSC_NULL_CHARACTER,
     &                    'density',0,0,300,300,view0,ierr)
      call PetscViewerDrawOpen(PETSC_COMM_WORLD,PETSC_NULL_CHARACTER,
     &                    'velocity',320,0,300,300,view1,ierr)
      call PetscViewerDrawOpen(PETSC_COMM_WORLD,PETSC_NULL_CHARACTER,
     &                    'total energy',640,0,300,300,view2,ierr)
      call PetscViewerDrawOpen(PETSC_COMM_WORLD,PETSC_NULL_CHARACTER,
     &                    'temperature',0,360,300,300,view3,ierr)
      call PetscViewerDrawOpen(PETSC_COMM_WORLD,PETSC_NULL_CHARACTER,
     &                    'pressure',320,360,300,300,view4,ierr)
c
c  graph initial conditions
c
      call FormGraph(x,view0,view1,view2,view3,view4,ierr)
c
c  copy x into xold
c
      call VecCopy(x,ctx(4),ierr)
      call FormDt(snes,x,ctx,ierr)
c
c################################
c
c  TIME STEP LOOP BEGIN
c
c################################
c
      ndt = 0

   10 if ( (ndt .le. nstep) .and. ((time + 1.0d-10) .lt. tfinal) ) then

        if (debug) then
          write(*,*)
          write(*,*) 'start of time loop'
          write(*,*)
          write(*,*) 'ndt = ',ndt
          write(*,*) 'nstep = ',nstep
          write(*,*) 'time = ',time
          write(*,*) 'tfinal = ',tfinal
          write(*,*)
        endif

        ndt = ndt + 1
c
c  increment time
c
        time = time + dt
        plotim = plotim + dt
c
c  call the nonlinear solver
c
        call SNESSolve(snes,PETSC_NULL_OBJECT,x,ierr) 
        CALL SNESGetIterationNumber(snes,its,ierr)
c
c  get the number of linear iterations used by the nonlinear solver
c
        call SNESGetNumberLinearIterations(snes,lits,ierr)

        if (debug) then
           write(*,*) 'in radhyd ',ndt,'x'
           call VecView(x,PETSC_VIEWER_STDOUT_SELF,ierr)
        endif
c
c  increment the counters
c
        totits = totits + its
        totlits = totlits + lits
c
c  Compute new time step
c
          call FormDt(snes,x,ctx,ierr)
c
c  Draw contour plot of solution
c
        if ( (mod(ndt,ngraph) .eq. 0) .or. (plotim .gt. tplot) )then
 
           plotim = plotim - tplot


        if (rank .eq. 0) then
           write(6,100) totits,totlits,ndt,dt,time
        endif
  100   format('Newt = ',i7,' lin =',i7,' step =',i7,
     &         ' dt = ',e8.3,' time = ',e10.4)
c
c  graph state conditions
c
          call FormGraph(x,view0,view1,view2,view3,view4,ierr)

        endif
c
c copy x into xold
c
        call VecCopy(x,ctx(4),ierr)


        goto 10

      endif
c
c################################
c
c  TIME STEP LOOP END
c
c################################
c

c
c  graph final conditions
c
      call FormGraph(x,view0,view1,view2,view3,view4,ierr)


      write(*,*)
      write(*,*)
      write(*,*) 'total Newton iterations = ',totits
      write(*,*) 'total linear iterations = ',totlits
      write(*,*) 'Average Newton per time step = ',
     &                       dble(totits)/dble(ndt)
      write(*,*) 'Average Krylov per Newton = ',
     &                    dble(totlits)/dble(totits)
      write(*,*)
      write(*,*)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


      call MatDestroy(ctx(2),ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(ctx(4),ierr)
      call VecDestroy(r,ierr)
      call SNESDestroy(snes,ierr)
      call PetscViewerDestroy(view0,ierr)
      call PetscViewerDestroy(view1,ierr)
      call PetscViewerDestroy(view2,ierr)
      call PetscViewerDestroy(view3,ierr)
      call PetscViewerDestroy(view4,ierr)

      call PCDestroy(ctx(1),ierr)

      call PetscFinalize(ierr)

      close(87)

      stop
      end
      subroutine ApplicationDampit(x,deltx,w,ierr)
! ---------------------------------------------------------------------
!
!  ApplicationDampit - Damps the newton update, called by
!  the higher level routine FormDampit().
!
!  Input Parameter:
!  x    - current iterate
!  deltx   - update
!
!  Output Parameters:
!  x    - new iterate
!  ierr - error code 
!
!  Notes:
!  This routine only damps the density.  May want to add energy
!  in the future
!

      implicit none

!  Common blocks:
#include "comd.h"

!  Input/output variables:
      PetscScalar x(mx*neq),deltx(mx*neq),
     1            w(mx*neq)
      integer  ierr

!  Local variables:
      double precision facmin, fac, newx, xmin, se, dse
      double precision u,en,rn,run
      integer  i, jr, jru, je

      ierr   = 0

      if (debug) then
        write(*,*) 'begin damping'
        do i = 1,mx*neq
          write(*,*)'x(',i,') = ',x(i)
        enddo
        write(*,*)
        do i = 1,mx*neq
          write(*,*)'deltx(',i,') = ',deltx(i)
        enddo
      endif

      facmin = 1.0d+0
c
c  set the scale factor
c
      do i=1,mx
c
c  set pointers
c
        jr  = (neq*i) - 2
        jru = (neq*i) - 1
        je  = (neq*i)
c
c  make sure dencity stayes positive
c
        newx = x(jr) - deltx(jr)
        xmin = damfac * x(jr)

        if (newx .lt. xmin) then
          fac = (1.0d+0 - damfac)*x(jr)/deltx(jr)
          if (fac .lt. facmin) then
            if (debug) then
              write(*,*) 'density', i, damfac,facmin,fac,x(jr),deltx(jr)
            endif
            facmin = fac
          endif
        endif
c
c  make sure Total energy stayes positive
c
        newx = x(je) - deltx(je)
        xmin = damfac * x(je)

        if (newx .lt. xmin) then
          fac = (1.0d+0 - damfac)*x(je)/deltx(je)
          if (fac .lt. facmin) then
            if (debug) then
              write(*,*) 'energy T',i, damfac,facmin,fac,x(je),deltx(je)
            endif
            facmin = fac
          endif
        endif
c
c  make sure specific internal  energy stayes positive
c
 
        u = x(jru)/x(jr)
        se = (x(je)/x(jr)) - (0.5d+0 * u * u)

        en = x(je) - deltx(je)
        rn = x(jr) - deltx(jr)
        run = x(jru) - deltx(jru)

        dse = se - ( (en/rn) - (0.5d+0 * (run/rn) * (run/rn)) )


        newx = se - dse
        xmin = damfac * se

        if (newx .lt. xmin) then
          fac = (1.0d+0 - damfac) * se / dse
          if (fac .lt. facmin) then
            if (debug) then
              write(*,*) 'se',i, damfac,facmin,fac,se,dse
            endif
            facmin = fac
          endif
        endif

      enddo
c
c write out warning
c
      if (facmin .lt. 1.0d+0) then
        write(*,*) 'facmin = ',facmin, damfac,time
c
c  scale the vector
c
        do i=1,neq*mx
          w(i) = x(i) - (facmin * deltx(i))
        enddo
      else
        do i=1,neq*mx
          w(i) = x(i) -  deltx(i)
        enddo
      endif

      if (debug) then
        write(*,*) 'end damping'
        do i = 1,mx*neq
           write(*,*) 'w(',i,') = ',w(i)
        enddo
      endif

      return
      end
      subroutine ApplicationDt(x,xold,ierr)
! ---------------------------------------------------------------------
!
!  ApplicationDt - compute CFL numbers. Called by
!  the higher level routine FormDt().
!
!  Input Parameter:
!  x    - local vector data
!
!  Output Parameters:
!  ierr - error code 
!
!  Notes:
!  This routine uses standard Fortran-style computations over a 2-dim 
array.
!

      implicit none

!  Common blocks:
#include "comd.h"
#include "tube.h"

!  Input/output variables:
      PetscScalar   x(mx*neq), xold(mx*neq)
      integer  ierr

!  Local variables:
      integer  i, jr, jru, je
c
c new
c
      double precision rhoee,  rhoe,  rhop,  rhow,  rhoww,
     &                 rhouee, rhoue, rhoup, rhouw, rhouww,
     &                 ergee,  erge,  ergp,  ergw,  ergww,
     &                         vele,  velp,  velw
      double precision pressp,sndp, vrad, vradn, vradd, tcfl, hcfl
      double precision tcflg, hcflg, dtt, dth
      double precision te, tp, tw
      double precision ue, up, uw
      double precision see, sep, sew
c
c old
c
      double precision rhooee,  rhooe,  rhoop,  rhoow,  rhooww
      double precision rhouoee, rhouoe, rhouop, rhouow, rhouoww
      double precision ergoee,  ergoe,  ergop,  ergow,  ergoww
      double precision veloe,  velop,  velow
      double precision uop, seop, top
      double precision dtold, dttype
c
c  functions
c
      double precision eos

      dtold = dt

      ierr   = 0

      if (debug) then
        write(*,*) 'in start dt'
        do i = 1,mx*neq
          write(*,*)'x(',i,') = ',x(i)
        enddo
        write(*,*) 'tfinal = ',tfinal
        write(*,*) 'time = ',time
        write(*,*) 'dt = ',dt
        write(*,*) 'dtmax = ',dtmax
      endif

      sndp = -1.0d+20
      vradn = 0.0d+0
      vradd = 0.0d+0

c
c################################
c
c  loop over all cells begin
c
c################################
c
      do i=1,mx
c
c  set pointers
c
        jr  = (neq*i) - 2
        jru = (neq*i) - 1
        je  = (neq*i)
c
c
c  set scalars
c
        call Setpbc(i,x,
     &             rhoee,  rhoe,  rhop,  rhow,  rhoww,
     &             rhouee, rhoue, rhoup, rhouw, rhouww,
     &             ergee,  erge,  ergp,  ergw,  ergww,
     &                     vele,  velp,  velw)
c
c compute temperatures
c
        uw = rhouw / rhow
        up = rhoup / rhop
        ue = rhoue / rhoe

        see = (erge/rhoe) - (0.5d+0 * ue * ue)
        sep = (ergp/rhop) - (0.5d+0 * up * up)
        sew = (ergw/rhow) - (0.5d+0 * uw * uw)

        te  = see / csubv
        tp  = sep / csubv
        tw  = sew / csubv
c
c compute old temperature
c

        call Setpbc(i,xold,
     &             rhooee,  rhooe,  rhoop,  rhoow,  rhooww,
     &             rhouoee, rhouoe, rhouop, rhouow, rhouoww,
     &             ergoee,  ergoe,  ergop,  ergow,  ergoww,
     &                      veloe,  velop,  velow)

        call Setpbcn(rhooee,  rhooe,  rhoop,  rhoow,  rhooww,
     &             rhouoee, rhouoe, rhouop, rhouow, rhouoww,
     &             ergoee,  ergoe,  ergop,  ergow,  ergoww,
     &                      veloe,  velop,  velow,         i)

        uop = rhouop / rhoop

        seop = (ergop/rhoop) - (0.5d+0 * uop * uop)

        top  = seop / csubv
c
c  compute thermal cfl
c
        vradn = vradn + (abs(tp - top)/dt)
        vradd = vradd + (abs(te - tw) / (2.0d+0 * dx) )
c
c  compute hydro cfl
c

        pressp  = eos(rhop, rhoup, ergp)
        sndp = max(sndp,sqrt( (gamma*pressp) / rhop ))

      enddo
c
c################################
c
c  loop over all cells end
c
c################################
c

      vrad = vradn / vradd

      tcfl = (vrad * dt) / dx
      hcfl = (sndp * dt) / dx

      dtt = max(dx/vrad,1.0d-7)
      dtt = tcscal * dtt

      dth = hcscal * dx / sndp

      if (.not. dtcon) then
        dt = min (dth,dtt,dt*dtgrow)
        if (dt .lt. dtmin) then
           dt = dtmin
        endif
        if (dt .gt. dtmax) then
           dt = dtmax
        endif
        if ( (time + dt) .gt. tfinal) then
           dt = tfinal - time
        endif

        if (dt .eq. dth) then
           dttype = 1.0d+0
        elseif (dt .eq. dtt) then
           dttype = 2.0d+0
        elseif (dt .eq. dtold*dtgrow) then
           dttype = 3.0d+0
        elseif (dt .eq. dtmax) then
           dttype = 4.0d+0
        elseif (dt .eq. dtmin) then
           dttype = 5.0d+0
        elseif (dt .eq. tfinal - time) then
           dttype = 6.0
        else
           dttype = -1.0d+0
        endif

      endif
 
 
      write(87,1000) time,dt,dth/hcscal,dtt/tcscal
      write(88,1000) time,dttype

 1000 format(4(2x,e18.9))

      if (debug) then
        write(*,*) 'thermal cfl = ',tcfl,'hydro cfl = ',hcfl
        write(*,*) 'dtt = ',dtt,' dth = ',dth
        write(*,*) 'tfinal = ',tfinal
        write(*,*) 'time = ',time
        write(*,*) 'dt = ',dt
        write(*,*) 'dtmax = ',dtmax
        write(*,*) 
      endif

      return
      end
      subroutine ApplicationExact(x,ierr)
! ---------------------------------------------------------------------
!
!  ApplicationExact - Computes exact solution, called by
!  the higher level routine FormExact().
!
!  Input Parameter:
!  x - local vector data
!
!  Output Parameters:
!  x -    initial conditions
!  ierr - error code 
!
!  Notes:
!  This routine uses standard Fortran-style computations over a 1-dim 
array.
!

      implicit none

!  Common blocks:

#include "comd.h"

!  Input/output variables:
      PetscScalar  x(mx)
      integer ierr

!  Local variables:
      integer  i
      double precision xloc
      PetscScalar rexact


!  Set parameters

      ierr   = 0

      do i = 1,mx

        xloc = xl0 + (dble(i) * dx)
        x(i) = rexact(xloc,time)

      enddo

      return 
      end
      subroutine ApplicationFunction(x,f,xold,ierr)
! ---------------------------------------------------------------------
!
!  ApplicationFunction - Computes nonlinear function, called by
!  the higher level routine FormFunction().
!
!  Input Parameter:
!  x    - local vector data
!
!  Output Parameters:
!  f    - local vector data, f(x)
!  ierr - error code 
!
!  Notes:
!  This routine uses standard Fortran-style computations over a 2-dim 
array.
!

      implicit none

!  Common blocks:
#include "comd.h"

!  Input/output variables:
      PetscScalar   x(mx*neq),f(mx*neq),
     1              xold(mx*neq)
      integer       ierr

!  Local variables:
      integer  i, jr, jru, je
      double precision rhoee,  rhoe,  rhop,  rhow,  rhoww,
     &                 rhouee, rhoue, rhoup, rhouw, rhouww,
     &                 ergee,  erge,  ergp,  ergw,  ergww,
     &                         vele,  velp,  velw

      double precision cont, energy, mom

      ierr   = 0

      if (debug) then
        write(*,*) 'in function'
        do i = 1,mx*neq
          write(*,*)'x(',i,') = ',x(i)
        enddo
      endif
c
c################################
c
c  loop over all cells begin
c
c################################
c
      do i=1,mx
c
c  set pointers
c
      jr  = (neq*i) - 2
      jru = (neq*i) - 1
      je  = (neq*i)
c
c
c  set scalars
c
      call Setpbc(i,x,
     &             rhoee,  rhoe,  rhop,  rhow,  rhoww,
     &             rhouee, rhoue, rhoup, rhouw, rhouww,
     &             ergee,  erge,  ergp,  ergw,  ergww,
     &                     vele,  velp,  velw)
c
c  compute functions
c

       f(jr) = cont(rhoee,  rhoe,  rhop,  rhow,  rhoww,
     &              rhouee, rhoue, rhoup, rhouw, rhouww,
     &              ergee,  erge,  ergp,  ergw,  ergww,
     &                                         i,xold)


       f(jru) = mom(rhoee,  rhoe,  rhop,  rhow,  rhoww,
     &              rhouee, rhoue, rhoup, rhouw, rhouww,
     &              ergee,  erge,  ergp,  ergw,  ergww,
     &                                          i,xold)


       f(je) = energy(rhoee,  rhoe,  rhop,  rhow,  rhoww,
     &                rhouee, rhoue, rhoup, rhouw, rhouww,
     &                ergee,  erge,  ergp,  ergw,  ergww,
     &                                            i,xold)

       if (debug) then
         write(*,*) 
         write(*,*) i,jr,jru,je,'res,r,ru,e'
         write(*,*) f(jr),f(jru),f(je)
         write(*,*) 
       endif

      enddo
c
c################################
c
c  loop over all cells end
c
c################################
c

      if (debug) then
        write(*,*) 'in function'
        do i = 1,mx*neq
           write(*,*) 'f(',i,') = ',f(i)
        enddo
      endif

      return
      end
      subroutine ApplicationInitialGuess(x,ierr)
! ---------------------------------------------------------------------
!
!  ApplicationInitialGuess - Computes initial approximation, called by
!  the higher level routine FormInitialGuess().
!
!  Input Parameter:
!  x - local vector data
!
!  Output Parameters:
!  x -    initial conditions
!  ierr - error code 
!
!  Notes:
!  This routine uses standard Fortran-style computations over a 1-dim 
array.
!

      implicit none

!  Common blocks:

#include "comd.h"
#include "tube.h"

!  Input/output variables:
      PetscScalar  x(mx*neq)
      integer ierr

!  Local variables:
      integer  i, j, jr, jru, je
      double precision xloc, re, ee, ve
      double precision wid, efloor
      PetscScalar uexact, rexact, eexact


CVAM  efloor = max(1.0d-1,1.0d-3 * erg0)
      efloor = 1.0d-1
CVAM  wid = max(1.0d-2,dx)
      wid = 1.0d-2

!  Set parameters

      ierr   = 0

      do i = 1,mx

        jr  = (neq*i) - 2
        jru = (neq*i) - 1
        je  = (neq*i)

        xloc = xl0 + (dble(i) * dx)

        if (probnum .eq. 1) then
           re = rexact(xloc,zero)
           ve = uexact(xloc,zero)
           ee = eexact(xloc,zero)
        else
           re = 1.0d+0
           ve = 0.0d+0
           ee = efloor + (erg0 * exp(-(xloc*xloc)/(wid*wid)))
        endif

        x(jr)  = re
        x(jru) = re * ve
        x(je)  = re * ( (0.5d+0 * ve * ve) + ee )

        if (debug) then
           write(*,100) i,jr,jru,je,xloc,x(jr),x(jru),x(je)
 100       format(i3,2x,i3,2x,i3,2x,i3,4(2x,e12.5))
        endif

      enddo

      call exact0
      call eval2
      call rval2
      call wval
      call uval2
      v3 = v2
      call val3

      a1 = sqrt(gamma*p1/r1)
      a2 = sqrt(gamma*p2/r2)
      a3 = sqrt(gamma*p3/r3)
      a4 = sqrt(gamma*p4/r4)

      write(*,1000) r1,r2,r3,r4
      write(*,2000) p1,p2,p3,p4
      write(*,3000) e1,e2,e3,e4
      write(*,4000) a1,a2,a3,a4
      write(*,*)

 1000 format ('rhos      ',4(f12.6))
 2000 format ('pressures ',4(f12.6))
 3000 format ('energies  ',4(f12.6))
 4000 format ('sound     ',4(f12.6))


      return 
      end
      subroutine ApplicationXmgr(x,ivar,ierr)
! ---------------------------------------------------------------------
!
!  ApplicationXmgr - Sets the Xmgr output called from
!  the higher level routine FormXmgr().
!
!  Input Parameter:
!  x - local vector data
!
!  Output Parameters:
!  x -    initial conditions
!  ierr - error code 
!
!  Notes:
!  This routine uses standard Fortran-style computations over a 1-dim 
array.
!

      implicit none

!  Common blocks:

#include "comd.h"

!  Input/output variables:
      PetscScalar  x(mx)
      integer ivar,ierr

!  Local variables:
      integer  i
      double precision xloc, sum
      PetscScalar rexact
      integer iplotnum(5)
      save iplotnum
      character*8 grfile


      data iplotnum / -1,-1,-1,-1,-1 /



!  Set parameters

      iplotnum(ivar) = iplotnum(ivar) + 1
      ierr   = 0

      if (ivar .eq. 1) then
         write(grfile,4000) iplotnum(ivar)
 4000    format('Xmgrr',i3.3)
      elseif (ivar .eq. 2) then
         write(grfile,5000) iplotnum(ivar)
 5000    format('Xmgru',i3.3)
      elseif (ivar .eq. 3) then
         write(grfile,6000) iplotnum(ivar)
 6000    format('Xmgre',i3.3)
      elseif (ivar .eq. 4) then
         write(grfile,7000) iplotnum(ivar)
 7000    format('Xmgrt',i3.3)
      else
         write(grfile,8000) iplotnum(ivar)
 8000    format('Xmgrp',i3.3)
      endif

      open (unit=44,file=grfile,status='unknown')

      do i = 1,mx

        xloc = xl0 + (dble(i) * dx)
        if ( (ivar .eq. 1) .and. (probnum .eq. 1) ) then
          write(44,1000) xloc, x(i), rexact(xloc,time)
        else
          write(44,1000) xloc, x(i)
        endif

      enddo

 1000 format(3(e18.12,2x))
      close(44)

      if ( (ivar .eq. 1) .and. (probnum .eq. 1) ) then
        sum = 0.0d+0
        do i = 1,mx
           xloc = xl0 + (dble(i) * dx)
           sum = sum + (x(i) - rexact(xloc,time)) ** 2
        enddo
        sum = sqrt(sum)

        write(*,*) 
        write(*,*)  'l2norm of the density error is',sum
        write(*,*) 
      endif


      return 
      end
      subroutine FormDampit(snes,ctx,x,f,g,y,w,
     &                       fnorm,ynorm,gnorm,flag,ierr)
! ---------------------------------------------------------------------
!
!  FormDampit - damps the Newton update
!
!  Input Parameters:
!  snes  - the SNES context
!  x     - current iterate
!  f     - residual evaluated at x
!  y     - search direction (containes new iterate on output)
!  w     - work vector
!  fnorm - 2-norm of f
!
!  In this example the application context is a Fortran integer array:
!      ctx(1) = shell preconditioner pressure matrix contex
!      ctx(2) = semi implicit pressure matrix
!      ctx(4) = xold  - old time values need for time advancement
!      ctx(5) = mx    - number of control volumes
!      ctx(6) = N     - total number of unknowns
!
!  Output Parameter:
!  g     - residual evaluated at new iterate y
!  y     - new iterate (contains search direction on input
!  gnorm - 2-norm of g
!  ynorm - 2-norm of search length
!  flag  - set to 0 if the line search succeeds; -1 on failure
!
!  Notes:
!  This routine serves as a wrapper for the lower-level routine
!  "ApplicationDampit", where the actual computations are 
!  done using the standard Fortran style of treating the local
!  vector data as a multidimensional array over the local mesh.
!  This routine merely accesses the local vector data via
!  VecGetArray() and VecRestoreArray().
!
      implicit none

#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscsnes.h"

!  Input/output variables:
      SNES             snes
      Vec              x, f, g, y, w
      PetscFortranAddr ctx(*)
      PetscScalar           fnorm, ynorm, gnorm
      integer          ierr, flag

!  Common blocks:

#include "comd.h"

!  Local variables:

!  Declarations for use with local arrays:
      PetscScalar lx_v(0:1),ly_v(0:1),
     1            lw_v(0:1)
      PetscOffset lx_i,ly_i,lw_i

c
c  set ynorm
c
      call VecNorm(y,NORM_2,ynorm,ierr)
c
c  copy x to w
c
      call VecCopy(x,w,ierr)
c
c get pointers to x, y, w
c

      call VecGetArray(x,lx_v,lx_i,ierr)
      call VecGetArray(y,ly_v,ly_i,ierr)
      call VecGetArray(w,lw_v,lw_i,ierr)
c
c  Compute Damping 
c
      call ApplicationDampit(lx_v(lx_i),ly_v(ly_i),lw_v(lw_i),ierr)
c
c  Restore vectors x, y, w
c
      call VecRestoreArray(x,lx_v,lx_i,ierr)
      call VecRestoreArray(y,ly_v,ly_i,ierr)
      call VecRestoreArray(w,lw_v,lw_i,ierr)
c
c  copy w to y
c
      call VecCopy(w,y,ierr)
c
c  compute new residual
c
      call SNESComputeFunction(snes,y,g,ierr)
      call VecNorm(g,NORM_2,gnorm,ierr)
      flag = 0

      if (debug) then
         write(*,*) 'form damp ynorm = ',ynorm
         write(*,*) 'gnorm = ',gnorm
      endif

      return 
      end
      subroutine FormDt(snes,x,ctx,ierr)
! ---------------------------------------------------------------------
!
!  FormDt - Compute CFL numbers
!
!  Input Parameters:
!  snes  - the SNES context
!  x     - input vector
!
!  In this example the application context is a Fortran integer array:
!      ctx(1) = shell preconditioner pressure matrix contex
!      ctx(2) = semi implicit pressure matrix
!      ctx(4) = xold  - old time values need for time advancement
!      ctx(5) = mx    - number of control volumes
!      ctx(6) = N     - total number of unknowns
!
!
!  Notes:
!  This routine serves as a wrapper for the lower-level routine
!  "ApplicationDt", where the actual computations are 
!  done using the standard Fortran style of treating the local
!  vector data as a multidimensional array over the local mesh.
!  This routine merely accesses the local vector data via
!  VecGetArray() and VecRestoreArray().
!
      implicit none

#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscsnes.h"

!  Input/output variables:
      SNES             snes
      Vec              x
      PetscFortranAddr ctx(*)
      integer          ierr

!  Common blocks:

#include "comd.h"

!  Local variables:

!  Declarations for use with local arrays:
      PetscScalar      lx_v(0:1)
      PetscOffset lx_i
      PetscScalar      lxold_v(0:1)
      PetscOffset lxold_i 

c
c get pointers to x, xold
c

      call VecGetArray(x,lx_v,lx_i,ierr)
      call VecGetArray(ctx(4),lxold_v,lxold_i,ierr)
c
c  Compute function 
c
      call ApplicationDt(lx_v(lx_i),lxold_v(lxold_i),ierr)
c
c  Restore vectors x, xold
c
      call VecRestoreArray(x,lx_v,lx_i,ierr)
      call VecRestoreArray(ctx(4),lxold_v,lxold_i,ierr)

      return 
      end
      subroutine FormExact(x,ierr)
! ---------------------------------------------------------------------
!
!  FormExact - Forms exact solution
!
!  Input Parameter:
!  x - vector
!
!  Output Parameters:
!  x - vector
!  ierr - error code 
!
!  Notes:
!  This routine serves as a wrapper for the lower-level routine
!  "ApplicationExact", where the actual computations are 
!  done using the standard Fortran style of treating the local
!  vector data as a multidimensional array over the local mesh.
!  This routine merely accesses the local vector data via
!  VecGetArray() and VecRestoreArray().
!
      implicit none

#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscmat.h"
#include "finclude/petscsnes.h"

!  Input/output variables:
      Vec      x
      integer  ierr

!  Declarations for use with local arrays:
      PetscScalar      lx_v(0:1)
      PetscOffset lx_i

      ierr   = 0

c
c  get a pointer to x
c
      call VecGetArray(x,lx_v,lx_i,ierr)
c
c  Compute initial guess 
c
      call ApplicationExact(lx_v(lx_i),ierr)
c
c  Restore vector x
c
      call VecRestoreArray(x,lx_v,lx_i,ierr)

      return 
      end
      subroutine FormFunction(snes,x,f,ctx,ierr)
! ---------------------------------------------------------------------
!
!  FormFunction - Evaluates nonlinear function, f(x).
!
!  Input Parameters:
!  snes  - the SNES context
!  x     - input vector
!
!  In this example the application context is a Fortran integer array:
!      ctx(1) = shell preconditioner pressure matrix contex
!      ctx(2) = semi implicit pressure matrix
!      ctx(4) = xold  - old time values need for time advancement
!      ctx(5) = mx    - number of control volumes
!      ctx(6) = N     - total number of unknowns
!
!  Output Parameter:
!  f     - vector with newly computed function
!
!  Notes:
!  This routine serves as a wrapper for the lower-level routine
!  "ApplicationFunction", where the actual computations are 
!  done using the standard Fortran style of treating the local
!  vector data as a multidimensional array over the local mesh.
!  This routine merely accesses the local vector data via
!  VecGetArray() and VecRestoreArray().
!
      implicit none

#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscsnes.h"

!  Input/output variables:
      SNES             snes
      Vec              x, f
      PetscFortranAddr ctx(*)
      integer          ierr

!  Common blocks:

#include "comd.h"

!  Local variables:

!  Declarations for use with local arrays:
      PetscScalar      lx_v(0:1), lf_v(0:1)
      PetscOffset lx_i, lf_i 
      PetscScalar      lxold_v(0:1)
      PetscOffset lxold_i 

c
c get pointers to x, f, xold
c

      call VecGetArray(x,lx_v,lx_i,ierr)
      call VecGetArray(f,lf_v,lf_i,ierr)
      call VecGetArray(ctx(4),lxold_v,lxold_i,ierr)
c
c  Compute function 
c
      call ApplicationFunction(lx_v(lx_i),lf_v(lf_i),
     &                             lxold_v(lxold_i),ierr)
c
c  Restore vectors x, f, xold
c
      call VecRestoreArray(x,lx_v,lx_i,ierr)
      call VecRestoreArray(f,lf_v,lf_i,ierr)
      call VecRestoreArray(ctx(4),lxold_v,lxold_i,ierr)
c
c something to do with profiling
c
      call PetscLogFlops(11.0d0*mx,ierr)

      return 
      end
      subroutine FormGraph(x,view0,view1,view2,view3,view4,ierr)
! ---------------------------------------------------------------------
!
!  FormGraph - Forms Graphics output
!
!  Input Parameter:
!  x - vector
!  time - scalar
!
!  Output Parameters:
!  ierr - error code 
!
!  Notes:
!  This routine serves as a wrapper for the lower-level routine
!  "ApplicationXmgr", where the actual computations are 
!  done using the standard Fortran style of treating the local
!  vector data as a multidimensional array over the local mesh.
!  This routine merely accesses the local vector data via
!  VecGetArray() and VecRestoreArray().
!
      implicit none

#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscmat.h"
#include "finclude/petscsnes.h"

#include "comd.h"
#include "tube.h"

!  Input/output variables:
      Vec      x
      integer  ierr

!  Declarations for use with local arrays:
      IS                 rfrom,rto,
     1                   rufrom, ruto, efrom, eto
      Vec                rval
      Vec                uval
      Vec                ruval
      Vec                eval
      Vec                seval
      Vec                pval
      Vec                kval
      Vec                tval
      Vec                steval
      VecScatter         scatter
      PetscViewer        view0,view1,
     1                   view2, view3, view4
      double precision   minus1, l2err, gm1, csubvi


      csubvi = 1.0d+0 / csubv
      gm1 = gamma - 1.0d+0
      ierr   = 0
      minus1 = -1.0d+0
c
c  graphics vectors
c
      CALL VecCreate(PETSC_COMM_WORLD,rval,ierr)
      CALL VecSetSizes(rval,PETSC_DECIDE,mx,ierr)
      CALL VecSetType(rval,VECMPI,ierr)
      call VecSetFromOptions(rval,ierr)
      call VecDuplicate(rval,uval,ierr)
      call VecDuplicate(rval,ruval,ierr)
      call VecDuplicate(rval,eval,ierr)
      call VecDuplicate(rval,seval,ierr)
      call VecDuplicate(rval,pval,ierr)
      call VecDuplicate(rval,kval,ierr)
      call VecDuplicate(rval,tval,ierr)
      call VecDuplicate(rval,steval,ierr)
c
c create index sets for vector scatters
c
      call ISCreateStride(PETSC_COMM_WORLD,mx,0,neq,rfrom, ierr)
      call ISCreateStride(PETSC_COMM_WORLD,mx,0,1,  rto,   ierr)
      call ISCreateStride(PETSC_COMM_WORLD,mx,1,neq,rufrom,ierr)
      call ISCreateStride(PETSC_COMM_WORLD,mx,0,1,  ruto,  ierr)
      call ISCreateStride(PETSC_COMM_WORLD,mx,2,neq,efrom, ierr)
      call ISCreateStride(PETSC_COMM_WORLD,mx,0,1,  eto,   ierr)

c
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c
c
c  load rval from x
c
      call VecScatterCreate(x,rfrom,rval,rto,scatter,ierr)
      call VecScatterBegin(scatter,x,rval,INSERT_VALUES,
     &                 SCATTER_FORWARD,ierr)
      call VecScatterEnd(scatter,x,rval,INSERT_VALUES,
     &                 SCATTER_FORWARD,ierr)
      call VecScatterDestroy(scatter,ierr)
c
c  plot rval vector
c
      call VecView(rval,view0,ierr)
c
c  make xmgr plot of rval
c
      call FormXmgr(rval,1,ierr)
c
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c
c
c  load eval from x
c
      call VecScatterCreate(x,efrom,eval,eto,scatter,ierr)
      call VecScatterBegin(scatter,x,eval,INSERT_VALUES,
     &                 SCATTER_FORWARD,ierr)
      call VecScatterEnd(scatter,x,eval,INSERT_VALUES,
     &                 SCATTER_FORWARD,ierr)
      call VecScatterDestroy(scatter,ierr)
c
c  plot eval vector
c
      call VecView(eval,view2,ierr)
c
c  make xmgr plot of eval
c
      call FormXmgr(eval,3,ierr)
c
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c

c
c  load ruval from x
c
      call VecScatterCreate(x,rufrom,ruval,ruto,scatter,ierr)
      call VecScatterBegin(scatter,x,ruval,INSERT_VALUES,
     &                 SCATTER_FORWARD,ierr)
      call VecScatterEnd(scatter,x,ruval,INSERT_VALUES,
     &                 SCATTER_FORWARD,ierr)
      call VecScatterDestroy(scatter,ierr)
c
c  create u = ru / r
c
      call VecPointwiseDivide(ruval,rval,uval,ierr)
c
c  plot uval vector
c
      call VecView(uval,view1,ierr)
c
c  make xmgr plot of uval
c
      call FormXmgr(uval,2,ierr)

c
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c

      call VecPointwiseMult(kval,uval,uval,ierr)
      call VecScale(kval,0.5d+0,ierr)

      call VecPointwiseDivide(steval,eval,rval,ierr)
      call VecWAXPY(seval,-1.0d+0,kval,steval,ierr)

      call VecCopy(seval,tval,ierr)
      call VecScale(tval,csubvi,ierr)

c
c  plot tval vector
c
      call VecView(tval,view3,ierr)
c
c  make xmgr plot of tval
c
      call FormXmgr(tval,4,ierr)

c
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c

      call VecPointwiseMult(rval,seval,pval,ierr)
      call VecScale(pval,gm1,ierr)
c
c  plot pval vector
c
      call VecView(pval,view4,ierr)
c
c  make xmgr plot of pval
c
      call FormXmgr(pval,5,ierr)
c
c%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c





! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call VecDestroy(rval, ierr)
      call VecDestroy(uval, ierr)
      call VecDestroy(ruval,ierr)
      call VecDestroy(eval, ierr)
      call VecDestroy(seval, ierr)
      call VecDestroy(pval, ierr)
      call VecDestroy(kval, ierr)
      call VecDestroy(tval, ierr)
      call VecDestroy(steval, ierr)

      call ISDestroy(rfrom, ierr)
      call ISDestroy(rto,   ierr)

      call ISDestroy(rufrom,ierr)
      call ISDestroy(ruto,  ierr)

      call ISDestroy(efrom, ierr)
      call ISDestroy(eto,   ierr)


      return 
      end
      subroutine FormInitialGuess(x,ierr)
! ---------------------------------------------------------------------
!
!  FormInitialGuess - Forms initial approximation.
!
!  Input Parameter:
!  x - vector
!
!  Output Parameters:
!  x - vector
!  ierr - error code 
!
!  Notes:
!  This routine serves as a wrapper for the lower-level routine
!  "ApplicationInitialGuess", where the actual computations are 
!  done using the standard Fortran style of treating the local
!  vector data as a multidimensional array over the local mesh.
!  This routine merely accesses the local vector data via
!  VecGetArray() and VecRestoreArray().
!
      implicit none

#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscmat.h"
#include "finclude/petscsnes.h"

!  Input/output variables:
      Vec      x
      integer  ierr

!  Declarations for use with local arrays:
      PetscScalar      lx_v(0:1)
      PetscOffset lx_i

      ierr   = 0

c
c  get a pointer to x
c
      call VecGetArray(x,lx_v,lx_i,ierr)
c
c  Compute initial guess 
c
      call ApplicationInitialGuess(lx_v(lx_i),ierr)
c
c  Restore vector x
c
      call VecRestoreArray(x,lx_v,lx_i,ierr)

      return 
      end
      subroutine FormXmgr(x,ivar,ierr)
! ---------------------------------------------------------------------
!
!  FormXmgr - Forms Xmgr output
!
!  Input Parameter:
!  x - vector
!
!  Output Parameters:
!  x - vector
!  ierr - error code 
!
!  Notes:
!  This routine serves as a wrapper for the lower-level routine
!  "ApplicationXmgr", where the actual computations are 
!  done using the standard Fortran style of treating the local
!  vector data as a multidimensional array over the local mesh.
!  This routine merely accesses the local vector data via
!  VecGetArray() and VecRestoreArray().
!
      implicit none

#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscmat.h"
#include "finclude/petscsnes.h"

!  Input/output variables:
      Vec      x
      integer  ivar,ierr

!  Declarations for use with local arrays:
      PetscScalar      lx_v(0:1)
      PetscOffset lx_i

      ierr   = 0

c
c  get a pointer to x
c
      call VecGetArray(x,lx_v,lx_i,ierr)
c
c  make the graph
c
      call ApplicationXmgr(lx_v(lx_i),ivar,ierr)
c
c  Restore vector x
c
      call VecRestoreArray(x,lx_v,lx_i,ierr)

      return 
      end
      subroutine PCRadApply(pc,x,y,ierr)
! ------------------------------------------------------------------- 
!
!   PCRadApply - This routine demonstrates the use of a
!   user-provided preconditioner.
!
!   Input Parameters:
!   pc - preconditioner object
!   x - input vector
!  In this example the shell preconditioner application context
!  is a Fortran integer array:
!      ctx(1) = shell preconditioner pressure matrix contex
!      ctx(2) = semi implicit pressure matrix
!      ctx(4) = xold  - old time values need for time advancement
!      ctx(5) = mx    - number of control volumes
!      ctx(6) = N     - total number of unknowns
!
!   Output Parameters:
!   y - preconditioned vector
!   ierr  - error code (nonzero if error has been detected)
!
!   Notes:
!   This code implements the Jacobi preconditioner plus the 
!   SOR preconditioner
!

      implicit none

#include "finclude/petscsys.h"
#include "finclude/petscvec.h"

#include "comd.h"
c
c  Input
c
      PC               pc
      Vec              x, y
      integer          ierr
c
c  Local
c
      PetscFortranAddr ctx(*)
      IS               defrom, deto
      Vec              de, rese
      VecScatter       scatter
      PetscScalar      lde_v(0:1),lrese_v(0:1)
      PetscOffset      lde_i,     lrese_i

      call PCShellGetContext(pc,ctx,ierr)

c
c  Identity preconditioner
c
      call VecCopy(x,y,ierr)
c
c  if kappa0 not equal to zero then precondition the radiation diffusion
c
      if (kappa0 .ne. 0.0d+0) then
 

c
c  Create needed vectors
c
         CALL VecCreate(PETSC_COMM_WORLD,de,ierr)
         CALL VecSetSizes(de,PETSC_DECIDE,mx,ierr)
         CALL VecSetType(de,VECMPI,ierr)
         call VecSetFromOptions(de,ierr)
         call VecDuplicate(de,rese,ierr)
c
c  create index sets for scatters
c
         call ISCreateStride(PETSC_COMM_WORLD,mx,2,neq,defrom,ierr)
         call ISCreateStride(PETSC_COMM_WORLD,mx,0,1,deto,ierr)
c
c  load rese from x
c
         call VecScatterCreate(x,defrom,rese,deto,scatter,ierr)
         call VecScatterBegin(scatter,x,rese,INSERT_VALUES,
     &                 SCATTER_FORWARD,ierr)
         call VecScatterEnd(scatter,x,rese,INSERT_VALUES,
     &                 SCATTER_FORWARD,ierr)
         call VecScatterDestroy(scatter,ierr)
c
c  apply preconditioner
c
      call PCApply(ctx(1),rese,de,ierr)

      if (debug) then
        write(*,*) 'PCRadApply dh is'
        call VecView(de,PETSC_VIEWER_STDOUT_SELF,ierr)
      endif
c
c load de into y
c
      call VecScatterCreate(de,deto,y,defrom,scatter,ierr)
      call VecScatterBegin(scatter,de,y,INSERT_VALUES,
     &                 SCATTER_FORWARD,ierr)
      call VecScatterEnd(scatter,de,y,INSERT_VALUES,
     &                 SCATTER_FORWARD,ierr)
      call VecScatterDestroy(scatter,ierr)

      if (debug) then
        write(*,*) 'PCRadApply y is'
        call VecView(y,PETSC_VIEWER_STDOUT_SELF,ierr)
      endif

      call VecDestroy(de,ierr)
      call VecDestroy(rese,ierr)

      call ISDestroy(defrom,ierr)
      call ISDestroy(deto,ierr)

      endif


      return
      end
      subroutine PCRadSetUp(pc,ierr)
!
!   PCRadSetUp - This routine sets up a user-defined
!   preconditioner context. 
!
!   Input Parameters:
!  In this example the shell preconditioner application context
!  is a Fortran integer array:
!      ctx(1) = shell preconditioner pressure matrix contex
!      ctx(2) = semi implicit pressure matrix
!      ctx(4) = xold  - old time values need for time advancement
!      ctx(5) = mx    - number of control volumes
!      ctx(6) = N     - total number of unknowns
!
!   Output Parameter:
!   ierr  - error code (nonzero if error has been detected)
!
!   Notes:
!   In this example, we define the shell preconditioner to be Jacobi
!   method.  Thus, here we create a work vector for storing the reciprocal
!   of the diagonal of the preconditioner matrix; this vector is then
!   used within the routine PCRadApply().
!

      implicit none

#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscmat.h"

#include "comd.h"
c
c  Input
c
      PC               pc
      integer          ierr
c
c  Local
c
      PetscFortranAddr ctx(*)
      Vec              eold
 
      PetscScalar      le_v(0:1)
      PetscOffset le_i

      call PCShellGetContext(pc,ctx,ierr)
 
c
c  create vector
c
      CALL VecCreate(PETSC_COMM_WORLD,eold,ierr)
      CALL VecSetSizes(eold,PETSC_DECIDE,mx,ierr)
      CALL VecSetType(eold,VECMPI,ierr)
      call VecSetFromOptions(eold,ierr)
c
c set up the matrix based on xold
c
      call Setmat(ctx,ierr)
c
c  set up the preconditioner
c
      call PCDestroy(ctx(1),ierr)
      call PCCreate(PETSC_COMM_WORLD,ctx(1),ierr)
CVAM  call PCSetType(ctx(1),PCJACOBI,ierr)
      call PCSetType(ctx(1),PCLU,ierr)
!      call PCSetVector(ctx(1),eold,ierr)
      call PCSetOperators(ctx(1),ctx(2),ctx(2),
     &              DIFFERENT_NONZERO_PATTERN,ierr)
      call PCSetUp(ctx(1),ierr)

      call VecDestroy(eold,ierr)


      return
      end
      subroutine Setmat(ctx,ierr)

      implicit none

#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscmat.h"
!  Common blocks:
#include "comd.h"
#include "tube.h"

!  Input/output variables:
      PetscFortranAddr ctx(*)
      integer  ierr

!  Local variables:
      PetscScalar      lx_v(0:1)
      PetscOffset      lx_i

      double precision xmult, himh, hiph, diag, upper, lower
      double precision hi, hip1, him1
      double precision
     &             rhoee,  rhoe,  rhop,  rhow,  rhoww,
     &             rhouee, rhoue, rhoup, rhouw, rhouww,
     &             ergee,  erge,  ergp,  ergw,  ergww,
     &                     ue,    up,    uw
      double precision see, sep, sew, seef, sewf, tef, twf,
     &                 ref, rwf, kef, kwf, xmulte, xmultw
c
      integer  im, nx, ny
c
c     get pointers to xold
c 
      call VecGetArray(ctx(4),lx_v,lx_i,ierr)
 

c
c############################
c
c loop over all cells begin
c
c############################
c
      do im = 1,mx
c
c  set scalars
c 
         call Setpbc(im,lx_v(lx_i),
     &             rhoee,  rhoe,  rhop,  rhow,  rhoww,
     &             rhouee, rhoue, rhoup, rhouw, rhouww,
     &             ergee,  erge,  ergp,  ergw,  ergww,
     &                     ue,    up,    uw)
c
c  set diffusion coefficients
c
        see = (erge/rhoe) - (0.5d+0 * ue * ue)
        sep = (ergp/rhop) - (0.5d+0 * up * up)
        sew = (ergw/rhow) - (0.5d+0 * uw * uw)

        seef = 0.5d+0 * (see + sep)
        sewf = 0.5d+0 * (sew + sep)

        tef = seef / csubv
        twf = sewf / csubv

        ref = 0.5d+0 * (rhoe + rhop)
        rwf = 0.5d+0 * (rhow + rhop)

        kef = kappa0 * (ref ** kappaa) * (tef ** kappab)
        kwf = kappa0 * (rwf ** kappaa) * (twf ** kappab)

        if (wilson) then
           kef = 1.0d+0 / ((1.0d+0/kef)+(abs(see-sep)/(seef*dx)))
           kwf = 1.0d+0 / ((1.0d+0/kwf)+(abs(sep-sew)/(sewf*dx)))
        endif
c
c  set coefficients
c
         xmult = dt / (dx * dx * csubv)

         xmulte = xmult * kef
         xmultw = xmult * kwf

         upper = -(xmulte / rhoe)
         lower = -(xmultw / rhow)

         diag = 1.0d+0 + ( (xmulte + xmultw) / rhop )

c
c  load coefficients into the matrix
c
         call MatSetValues(ctx(2),1,im-1,1,im-1,diag,INSERT_VALUES,ierr)

         if (im .eq. 1) then
           call MatSetValues(ctx(2),1,im-1,1,im  ,upper,
     1                       INSERT_VALUES,ierr) 
         elseif (im .eq. mx) then
           call MatSetValues(ctx(2),1,im-1,1,im-2,lower,
     1                       INSERT_VALUES,ierr) 
         else
           call MatSetValues(ctx(2),1,im-1,1,im  ,upper,
     1                       INSERT_VALUES,ierr) 
           call MatSetValues(ctx(2),1,im-1,1,im-2,lower,
     1                       INSERT_VALUES,ierr) 
         endif


      enddo
c
c############################
c
c loop over all cells end
c
c############################
c
 
c
c  final load of matrix
c
      call MatAssemblyBegin(ctx(2),MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(ctx(2),MAT_FINAL_ASSEMBLY,ierr)

      if (debug) then
        call MatGetSize(ctx(2),nx,ny,ierr)
        write(*,*) 'in setup nx = ',nx,' ny = ',ny
        call MatView(ctx(2),PETSC_VIEWER_DRAW_WORLD,ierr)
      endif

      call VecRestoreArray (ctx(4),lx_v,lx_i,ierr)



      return
      end
      subroutine Setpbc(i,x,
     &             rhoee,  rhoe,  rhop,  rhow,  rhoww,
     &             rhouee, rhoue, rhoup, rhouw, rhouww,
     &             ergee,  erge,  ergp,  ergw,  ergww,
     &                     vele,  velp,  velw)

      implicit none

!  Common blocks:
#include "comd.h"

!  Input/output variables:
      PetscScalar   x(mx*neq)
      integer  i
      double precision rhoee,  rhoe,  rhop,  rhow,  rhoww
      double precision rhouee, rhoue, rhoup, rhouw, rhouww
      double precision ergee,  erge,  ergp,  ergw,  ergww
      double precision         vele,  velp,  velw

!  Local variables:
      integer  jr, jru, je

c
c  set pointers
c
      jr  = (neq*i) - 2
      jru = (neq*i) - 1
      je  = (neq*i)

      if (debug) then
        write(*,*)
        write(*,*) 'in Setpbc jr,jru,je = ',jr,jru,je
        write(*,*)
      endif
 
      if (i .eq. 1) then

        rhoee = x(jr+(2*neq))
        rhoe  = x(jr+neq)
        rhop  = x(jr)
        rhow  = x(jr)
        rhoww = x(jr)

        rhouee = x(jru+(2*neq))
        rhoue  = x(jru+neq)
        rhoup  = x(jru)
        rhouw  = x(jru)
        rhouww = x(jru)

        ergee = x(je+(2*neq))
        erge  = x(je+neq)
        ergp  = x(je)
        ergw  = x(je)
        ergww = x(je)

        velw = 0.0d+0
        velp = rhoup/rhop
        vele = rhoue/rhoe

      elseif (i .eq. 2) then

        rhoee = x(jr+(2*neq))
        rhoe  = x(jr+neq)
        rhop  = x(jr)
        rhow  = x(jr-neq)
        rhoww = x(jr-neq)

        rhouee = x(jru+(2*neq))
        rhoue  = x(jru+neq)
        rhoup  = x(jru)
        rhouw  = x(jru-neq)
        rhouww = x(jru-neq)

        ergee = x(je+(2*neq))
        erge  = x(je+neq)
        ergp  = x(je)
        ergw  = x(je-neq)
        ergww = x(je-neq)

        velw = rhouw/rhow
        velp = rhoup/rhop
        vele = rhoue/rhoe

      elseif (i .eq. mx-1) then

        rhoee = x(jr+neq)
        rhoe  = x(jr+neq)
        rhop  = x(jr)
        rhow  = x(jr-neq)
        rhoww = x(jr-(2*neq))

        rhouee = x(jru+neq)
        rhoue  = x(jru+neq)
        rhoup  = x(jru)
        rhouw  = x(jru-neq)
        rhouww = x(jru-(2*neq))

        ergee = x(je+neq)
        erge  = x(je+neq)
        ergp  = x(je)
        ergw  = x(je-neq)
        ergww = x(je-(2*neq))

        velw = rhouw/rhow
        velp = rhoup/rhop
        vele = rhoue/rhoe

      elseif (i .eq. mx) then

        rhoee = x(jr)
        rhoe  = x(jr)
        rhop  = x(jr)
        rhow  = x(jr-neq)
        rhoww = x(jr-(2*neq))

        rhouee = x(jru)
        rhoue  = x(jru)
        rhoup  = x(jru)
        rhouw  = x(jru-neq)
        rhouww = x(jru-(2*neq))

        ergee = x(je)
        erge  = x(je)
        ergp  = x(je)
        ergw  = x(je-neq)
        ergww = x(je-(2*neq))

        velw = rhouw/rhow
        velp = rhoup/rhop
        vele = 0.0d+0

      else

        rhoee = x(jr+(2*neq))
        rhoe  = x(jr+neq)
        rhop  = x(jr)
        rhow  = x(jr-neq)
        rhoww = x(jr-(2*neq))

        rhouee = x(jru+(2*neq))
        rhoue  = x(jru+neq)
        rhoup  = x(jru)
        rhouw  = x(jru-neq)
        rhouww = x(jru-(2*neq))

        ergee = x(je+(2*neq))
        erge  = x(je+neq)
        ergp  = x(je)
        ergw  = x(je-neq)
        ergww = x(je-(2*neq))

        velw = rhouw/rhow
        velp = rhoup/rhop
        vele = rhoue/rhoe

      endif

      if (debug) then
         write(*,*)
         write(*,*) 'in Setpbc ',i,jr,jru,je
         write(*,*) 'mx = ',mx
         write(*,*)
      endif


      return
      end
      subroutine Setpbcn(rhoee,  rhoe,  rhop,  rhow,  rhoww,
     &                   rhouee, rhoue, rhoup, rhouw, rhouww,
     &                   ergee,  erge,  ergp,  ergw,  ergww,
     &                           ue,    up,    uw,           jbc)

      implicit none

!  Common blocks:
#include "comd.h"

!  Input/output variables:
      integer  jbc
      double precision rhoee,  rhoe,  rhop,  rhow,  rhoww
      double precision rhouee, rhoue, rhoup, rhouw, rhouww
      double precision ergee,  erge,  ergp,  ergw,  ergww
      double precision         ue,    up,    uw

!  Local variables:

      if (jbc .eq. 1) then
         rhoww  = rhop
         rhow   = rhop
         rhouww = rhoup
         rhouw  = rhoup
         ergww  = ergp
         ergw   = ergp
         uw     = 0.0d+0
      elseif  (jbc .eq. 2) then
         rhoww  = rhow
         rhouww = rhouw
         ergww  = ergw
         uw     = rhouw / rhow
      else
         uw = rhouw / rhow
      endif

      if (jbc .eq. mx) then
         rhoee  = rhop
         rhoe   = rhop
         rhouee = rhoup
         rhoue  = rhoup
         ergee  = ergp
         erge   = ergp
         ue     = 0.0d+0
      elseif (jbc .eq. mx-1) then
         rhoee  = rhoe
         rhouee = rhoue
         ergee  = erge
         ue     = rhoue / rhoe
      else
         ue     = rhoue / rhoe
      endif

      up = rhoup / rhop

      if (debug) then
         write(*,*) 'in Setpbcn ',jbc, 'mx = ',mx
      endif


      return
      end
      double precision function cont
     &            (rhoee,  rhoe,  rhop,  rhow,  rhoww,
     &             rhouee, rhoue, rhoup, rhouw, rhouww,
     &             ergee,  erge,  ergp,  ergw,  ergww,
     &                                      jcont,xold)
c
c  This function computes the residual
c  for the 1-D continuity equation
c
c
      implicit none

      include 'comd.h'
      include 'tube.h'
c
c     input variables
c
      double precision rhoee,  rhoe,  rhop,  rhow,  rhoww
      double precision rhouee, rhoue, rhoup, rhouw, rhouww
      double precision ergee,  erge,  ergp,  ergw,  ergww
      double precision xold(mx*neq)
c
      integer jcont
c
c     local variables
c
      double precision theta1
      integer jr
c
c  new
c
      double precision velfw, velfe
      double precision vele,velp,velw
      double precision fluxe, fluxw
      double precision urhoe, urhow
      double precision source
c
c old
c
      double precision rhooee,  rhooe,  rhoop,  rhoow,  rhooww
      double precision rhouoee, rhouoe, rhouop, rhouow, rhouoww
      double precision ergoee,  ergoe,  ergop,  ergow,  ergoww
      double precision teoee, teoe, teop, teow, teoww, 
     &                 uoe, uoee, uop, uow, uoww
      double precision velfow, velfoe
      double precision veloe,velop,velow
      double precision fluxoe, fluxow
      double precision urhooe, urhoow
      double precision sourceo
c
c functions
c
      double precision godunov2
      double precision upwind, fluxlim
c
c
c ******************************************************************
c
c
c
      if (debug) then
        write(*,*)
        write(*,*) 'in cont',jcont,' ihod = ',ihod
        write(*,*) 'rhoee = ',rhoee, ' rhoe = ',rhoe
        write(*,*) 'rhop = ',rhop
        write(*,*) 'rhoww = ',rhoww, ' rhow = ',rhow
        write(*,*)
      endif

      jr = (neq*jcont) - 2

c########################
c
c      NEW
c
c########################

      call Setpbcn(rhoee,  rhoe,  rhop,  rhow,  rhoww,
     &             rhouee, rhoue, rhoup, rhouw, rhouww,
     &             ergee,  erge,  ergp,  ergw,  ergww,
     &                     vele,  velp,  velw,         jcont)

      velfe = 0.5d+0 * (vele + velp)
      velfw = 0.5d+0 * (velw + velp)

      if (ihod .eq. 1) then

        urhoe = upwind(rhop,rhoe,velfe)
        urhow = upwind(rhow,rhop,velfw)

      elseif (ihod .eq. 2) then

        urhoe = fluxlim(rhow,rhop,rhoe,rhoee,velfe)
        urhow = fluxlim(rhoww,rhow,rhop,rhoe,velfw)

      endif

      if (ihod .eq. 3) then
        fluxe = (dt/dx) * godunov2(rhow, rhop, rhoe, rhoee,
     &                             rhouw,rhoup,rhoue,rhouee,
     &                             ergw, ergp, erge, ergee,1)
        fluxw = (dt/dx) * godunov2(rhoww, rhow, rhop, rhoe,
     &                             rhouww,rhouw,rhoup,rhoue,
     &                             ergww, ergw, ergp, erge,1)
      else
        fluxe = (dt/dx) * urhoe 
        fluxw = (dt/dx) * urhow 
      endif


      source = 0.0d+0

c########################
c
c      OLD
c
c########################

      call Setpbc(jcont,xold,
     &             rhooee,  rhooe,  rhoop,  rhoow,  rhooww,
     &             rhouoee, rhouoe, rhouop, rhouow, rhouoww,
     &             ergoee,  ergoe,  ergop,  ergow,  ergoww,
     &                      veloe,  velop,  velow)

      call Setpbcn(rhooee,  rhooe,  rhoop,  rhoow,  rhooww,
     &             rhouoee, rhouoe, rhouop, rhouow, rhouoww,
     &             ergoee,  ergoe,  ergop,  ergow,  ergoww,
     &                      veloe,  velop,  velow,         jcont)

      velfoe = 0.5d+0 * (veloe + velop)
      velfow = 0.5d+0 * (velow + velop)


      if (ihod .eq. 1) then

        urhooe = upwind(rhoop,rhooe,velfoe)
        urhoow = upwind(rhoow,rhoop,velfow)

      elseif (ihod .eq. 2) then

        urhooe = fluxlim(rhoow,rhoop,rhooe,rhooee,velfoe)
        urhoow = fluxlim(rhooww,rhoow,rhoop,rhooe,velfow)

      endif

      if (ihod .eq. 3) then
        fluxoe = (dt/dx) * godunov2(rhoow, rhoop, rhooe, rhooee,
     &                              rhouow,rhouop,rhouoe,rhouoee,
     &                              ergow, ergop, ergoe, ergoee,1)
        fluxow = (dt/dx) * godunov2(rhooww, rhoow, rhoop, rhooe,
     &                              rhouoww,rhouow,rhouop,rhouoe,
     &                              ergoww, ergow, ergop, ergoe,1)
      else
        fluxoe = (dt/dx) * urhooe
        fluxow = (dt/dx) * urhoow 
      endif

      sourceo = 0.0d+0


c########################
c
c      FUNCTION
c
c########################

      theta1 = 1.0d+0 - theta
      cont =  (rhop - xold(jr)) 
     &    + (  theta  * ( (fluxe  - fluxw ) - source  )  )
     &    + (  theta1 * ( (fluxoe - fluxow) - sourceo )  )
CVAM
      if (probnum .eq. 3) then
        cont = 0.0d+0
      endif
CVAM


      if (debug) then
       write(*,*)
       write(*,*) 'cont(',jcont,') = ',cont
       write(*,*) 'theta = ',theta,'rhop = ',rhop
       write(*,*) 'source = ',source,' sourceo = ',sourceo
       write(*,*) 'fluxe = ',fluxe,' fluxw = ',fluxw
       write(*,*) 'fluxoe = ',fluxoe,' fluxow = ',fluxow
       write(*,*) 'urhoe = ',urhoe,' urhow = ',urhow
       write(*,*) 'urhooe = ',urhooe,' urhoow = ',urhoow
       write(*,*)
      endif

      return
      end
      double precision function  eexact(x,t)

      implicit none

      double precision x,t
      double precision xot, head, tail, contact, ufan
      double precision xpow, grat, urat
      double precision uexact


      logical debug

      include 'tube.h'

      debug = .false.


      if (t .le. 0.0d+0) then
        if (x .gt. 0.0d+0) then
          eexact = e1
        else
          eexact = e4
        endif
      else

       xot = x/t
       head = -a4
       tail = v3 - a3
       contact = v2

       if (xot .lt. head) then
          eexact = e4
       elseif (xot .gt. sspd) then
          eexact = e1
       elseif (xot .gt. contact) then
          eexact = e2
       elseif (xot .gt. tail) then
          eexact = e3
       else
          ufan = uexact(x,t)
          grat = (gamma - 1.0d+0) / 2.0d+0
          xpow = 2.0d+0
          urat = ufan / a4
          eexact = e4 * (  ( 1.0d+0 - (grat * urat) ) ** xpow  )
       endif

      endif


      if (debug) then
        write(*,*)
        write(*,*) 'eexact(',x,',',t,') = ',eexact
        write(*,*)
      endif

      return
      end
      subroutine eigen(ht,uht)
c23456789012345678901234567890123456789012345678901234567890123456789012
c
c          subroutine eigen
c
c  This subroutine computes the eigen values and eigen vectors
c
c23456789012345678901234567890123456789012345678901234567890123456789012


c#######################################################################

      implicit none

      include 'comd.h'

      double precision ht, uht

      double precision ut, at, lam1, lam2


c#######################################################################

      ut = uht / ht
      at = sqrt( ht)

      lam1 = ut - at
      lam2 = ut + at

      eigval(1) = lam1
      eigval(2) = lam2

      eigvec(1,1) = 1.0d+0
      eigvec(2,1) = lam1
      eigvec(1,2) = 1.0d+0
      eigvec(2,2) = lam2

      rinv(1,1) =  lam2 / (2.0d+0 * at)
      rinv(2,1) = -lam1 / (2.0d+0 * at)
      rinv(1,2) = -1.0d+0 / (2.0d+0 * at)
      rinv(2,2) =  1.0d+0 / (2.0d+0 * at)


      return
      end
      subroutine eigene(r,ru,e,l1,l2,l3)
c23456789012345678901234567890123456789012345678901234567890123456789012
c
c          subroutine eigene
c
c  This subroutine computes the eigen values for the entropy fix
c
c23456789012345678901234567890123456789012345678901234567890123456789012


c#######################################################################

      implicit none

      include 'tube.h'

      double precision r,ru,e,l1,l2,l3

      double precision p,u,a

      double precision eos
      integer ierr

      logical debug


c#######################################################################

      debug = .false.

      if (debug) then
         write(*,*)
         write(*,*) 'gamma = ',gamma
         write(*,*) 'r,ru,e = ',r,ru,e
         write(*,*)
      endif

      p = eos(r,ru,e)
      u = ru/r
      if ( ((gamma * p)/r) .lt. 0.0d+0 ) then
         write(*,*) 
         write(*,*) 'gamma = ',gamma
         write(*,*) 'r = ',r
         write(*,*) 'p = ',p
         write(*,*) 
         call PetscFinalize(ierr)
         stop
      endif
      a = sqrt((gamma * p)/r)

      if (debug) then
         write(*,*)
         write(*,*) 'p,u,a = ',p,u,a
         write(*,*)
      endif

      l1 = u - a
      l2 = u
      l3 = u + a

      return
      end
      double precision function energy
     &               (rhoee,  rhoe,  rhop,  rhow,  rhoww,
     &                rhouee, rhoue, rhoup, rhouw, rhouww,
     &                ergee,  erge,  ergp,  ergw,  ergww,
     &                                      jerg,xold)
c
c  This function computes the residual
c  for the 1-D energy equation
c
c
      implicit none

      include 'comd.h'
      include 'tube.h'
c
c     input variables
c
      double precision rhoee,  rhoe,  rhop,  rhow,  rhoww
      double precision rhouee, rhoue, rhoup, rhouw, rhouww
      double precision ergee,  erge,  ergp,  ergw,  ergww
      double precision xold(mx*neq)
c
      integer jerg
c
c     local variables
c
      double precision theta1
      integer je
c
c  new
c
      double precision velfw, velfe
      double precision vele,velp,velw
      double precision fluxe, fluxw
      double precision uepe, uepw
      double precision ue, up, uw
      double precision see, sep, sew
      double precision seef, sewf
      double precision upe, upw
      double precision presse, pressw
      double precision source
      double precision te, tp, tw
      double precision tef, twf, ref, rwf
      double precision kef, kwf
      double precision hflxe, hflxw
c
c old
c
      double precision rhooee,  rhooe,  rhoop,  rhoow,  rhooww
      double precision rhouoee, rhouoe, rhouop, rhouow, rhouoww
      double precision ergoee,  ergoe,  ergop,  ergow,  ergoww
      double precision velfow, velfoe
      double precision veloe,velop,velow
      double precision fluxoe, fluxow
      double precision uepoe, uepow
      double precision uoe, uop, uow
      double precision seoe, seop, seow
      double precision seoef, seowf
      double precision upoe, upow
      double precision pressoe, pressow
      double precision sourceo
      double precision toe, top, tow
      double precision toef, towf, roef, rowf
      double precision koef, kowf
      double precision hflxoe, hflxow
c
c functions
c
      double precision godunov2, eos
      double precision upwind, fluxlim

c
c
c ******************************************************************
c
c
c
      je = (neq*jerg)

c########################
c
c      NEW
c
c########################

      call Setpbcn(rhoee,  rhoe,  rhop,  rhow,  rhoww,
     &             rhouee, rhoue, rhoup, rhouw, rhouww,
     &             ergee,  erge,  ergp,  ergw,  ergww,
     &                     vele,  velp,  velw,         jerg)

      pressw  = eos(rhow, rhouw, ergw)
      presse  = eos(rhoe, rhoue, erge)

      uw = rhouw / rhow
      up = rhoup / rhop
      ue = rhoue / rhoe

      upw = uw * pressw
      upe = ue * presse

      velfe = 0.5d+0 * (vele + velp)
      velfw = 0.5d+0 * (velw + velp)

      if (ihod .eq. 1) then

        uepe = upwind(ergp,erge,velfe)
        uepw = upwind(ergw,ergp,velfw)

      elseif (ihod .eq. 2) then

        uepe = fluxlim(ergw,ergp,erge,ergee,velfe)
        uepw = fluxlim(ergww,ergw,ergp,erge,velfw)

      endif

      if (ihod .eq. 3) then
        fluxe = (dt/dx) * godunov2(rhow, rhop, rhoe, rhoee,
     &                             rhouw,rhoup,rhoue,rhouee,
     &                             ergw, ergp, erge, ergee,3)
        fluxw = (dt/dx) * godunov2(rhoww, rhow, rhop, rhoe,
     &                             rhouww,rhouw,rhoup,rhoue,
     &                             ergww, ergw, ergp, erge,3)
      else
        fluxe = (dt/dx) * ( uepe  + (0.5d+0*upe) )
        fluxw = (dt/dx) * ( uepw  + (0.5d+0*upw) )
      endif
c
c  radiation
c
      if (kappa0 .eq. 0.0d+0) then
        source = 0.0d+0
      else

        see = (erge/rhoe) - (0.5d+0 * ue * ue)
        sep = (ergp/rhop) - (0.5d+0 * up * up)
        sew = (ergw/rhow) - (0.5d+0 * uw * uw)

        seef = 0.5d+0 * (see + sep)
        sewf = 0.5d+0 * (sew + sep)

        te  = see / csubv
        tp  = sep / csubv
        tw  = sew / csubv

        tef = seef / csubv
        twf = sewf / csubv

        ref = 0.5d+0 * (rhoe + rhop)
        rwf = 0.5d+0 * (rhow + rhop)

        kef = kappa0 * (ref ** kappaa) * (tef ** kappab)
        kwf = kappa0 * (rwf ** kappaa) * (twf ** kappab)

        if (wilson) then
           kef = 1.0d+0 / ((1.0d+0/kef)+(abs(see-sep)/(seef*dx)))
           kwf = 1.0d+0 / ((1.0d+0/kwf)+(abs(sep-sew)/(sewf*dx)))
        endif

        if ( debug .and. (kef .gt. 1.0d+10) ) then
          write(*,*) 'kef = ',kef,ref,tef,kappaa,kappab,kappa0
        endif
        if ( debug .and. (kwf .gt. 1.0d+10) ) then
          write(*,*) 'kwf = ',kwf,rwf,twf,kappaa,kappab,kappa0
        endif

        hflxe = kef * (te - tp) / dx
        hflxw = kwf * (tp - tw) / dx

        source = (dt/dx) * (hflxe - hflxw)

      endif

c########################
c
c      OLD
c
c########################

      call Setpbc(jerg,xold,
     &             rhooee,  rhooe,  rhoop,  rhoow,  rhooww,
     &             rhouoee, rhouoe, rhouop, rhouow, rhouoww,
     &             ergoee,  ergoe,  ergop,  ergow,  ergoww,
     &                      veloe,  velop,  velow)

      call Setpbcn(rhooee,  rhooe,  rhoop,  rhoow,  rhooww,
     &             rhouoee, rhouoe, rhouop, rhouow, rhouoww,
     &             ergoee,  ergoe,  ergop,  ergow,  ergoww,
     &                      veloe,  velop,  velow,         jerg)

      pressow  = eos(rhoow, rhouow, ergow)
      pressoe  = eos(rhooe, rhouoe, ergoe)

      uow = rhouow / rhoow
      uop = rhouop / rhoop
      uoe = rhouoe / rhooe

      upow = uow * pressow
      upoe = uoe * pressoe

      velfoe = 0.5d+0 * (veloe + velop)
      velfow = 0.5d+0 * (velow + velop)


      if (ihod .eq. 1) then

        uepoe = upwind(ergop,ergoe,velfoe)
        uepow = upwind(ergow,ergop,velfow)

      elseif (ihod .eq. 2) then

        uepoe = fluxlim(ergow,ergop,ergoe,ergoee,velfoe)
        uepow = fluxlim(ergoww,ergow,ergop,ergoe,velfow)

      endif

      if (ihod .eq. 3) then
        fluxoe = (dt/dx) * godunov2(rhoow, rhoop, rhooe, rhooee,
     &                              rhouow,rhouop,rhouoe,rhouoee,
     &                              ergow, ergop, ergoe, ergoee,3)
        fluxow = (dt/dx) * godunov2(rhooww, rhoow, rhoop, rhooe,
     &                              rhouoww,rhouow,rhouop,rhouoe,
     &                              ergoww, ergow, ergop, ergoe,3)
      else
        fluxoe = (dt/dx) * ( uepoe + (0.5d+0 * upoe) )
        fluxow = (dt/dx) * ( uepow + (0.5d+0 * upow) )
      endif

c
c  old radiation
c
      if (kappa0 .eq. 0.0d+0) then
        sourceo = 0.0d+0
      else

        seoe = (ergoe/rhooe) - (0.5d+0 * uoe * uoe)
        seop = (ergop/rhoop) - (0.5d+0 * uop * uop)
        seow = (ergow/rhoow) - (0.5d+0 * uow * uow)

        seoef = 0.5d+0 * (seoe + seop)
        seowf = 0.5d+0 * (seow + seop)

        toe  = seoe / csubv
        top  = seop / csubv
        tow  = seow / csubv

        toef = seoef / csubv
        towf = seowf / csubv

        roef = 0.5d+0 * (rhooe + rhoop)
        rowf = 0.5d+0 * (rhoow + rhoop)

        koef = kappa0 * (roef ** kappaa) * (toef ** kappab)
        kowf = kappa0 * (rowf ** kappaa) * (towf ** kappab)

        if (wilson) then
           koef = 1.0d+0 / ((1.0d+0/koef)+(abs(seoe-seop)/(seoef*dx)))
           kowf = 1.0d+0 / ((1.0d+0/kowf)+(abs(seop-seow)/(seowf*dx)))
        endif

        if ( debug .and. (koef .gt. 1.0d+10) ) then
          write(*,*) 'koef = ',koef,roef,toef,kappaa,kappab,kappa0
        endif
        if ( debug .and. (kowf .gt. 1.0d+10) ) then
          write(*,*) 'kowf = ',kowf,rowf,towf,kappaa,kappab,kappa0
        endif

        hflxoe = koef * (toe - top) / dx
        hflxow = kowf * (top - tow) / dx

        sourceo = (dt/dx) * (hflxoe - hflxow)

      endif


c########################
c
c      FUNCTION
c
c########################

CVAM
      if (probnum .eq. 3) then
        fluxe  = 0.0d+0
        fluxw  = 0.0d+0
        fluxoe = 0.0d+0
        fluxow = 0.0d+0
      endif
CVAM

      theta1 = 1.0d+0 - theta
      energy =  (ergp - xold(je)) 
     &    + (  theta  * ( (fluxe  - fluxw ) - source  )  )
     &    + (  theta1 * ( (fluxoe - fluxow) - sourceo )  )

      if (debug) then
        write(*,*)
        write(*,*) 'energy(',jerg,') = ',energy
        write(*,*)
        write(*,*) fluxe,fluxw
        write(*,*) fluxoe,fluxow
        write(*,*) source,sourceo
        write(*,*)
      endif

      return
      end
      double precision function eos(r,ru,e)

      implicit none

      double precision r,ru,e

      double precision se, u

      integer ierr

      logical debug

      include "tube.h"

      debug = .false.

      if (debug) then
        write(*,*) 
        write(*,*) 'in eos r,ru,e'
        write(*,*) r,ru,e
        write(*,*) 
      endif

      u = ru/r

      se = (e/r) - (0.5d+0 * u * u)
      eos = (gamma - 1.0d+0) * r * se

      if (eos .lt. 0.0d+0) then
         write(*,*)
         write(*,*) 'eos = ',eos
         write(*,*) 'gamma = ',gamma
         write(*,*) 'r = ',r
         write(*,*) 'se = ',se
         write(*,*) 'e = ',e
         write(*,*) 'u = ',u
         write(*,*) 'ru = ',ru
         call PetscFinalize(ierr)
         write(*,*)
         stop
      endif

      if (debug) then
        write(*,*) 
        write(*,*) 'in eos u,se,eos'
        write(*,*) u,se,eos
        write(*,*) 
      endif


      return
      end
      subroutine eval2

      implicit none

      double precision prat, grat, xnum, xdenom


      logical debug

      include 'tube.h'

      debug = .false.

      prat = p2/p1
      grat = (gamma + 1.0d+0) / (gamma - 1.0d+0)

      xnum = grat + prat
      xdenom = 1.0d+0 + (prat * grat)
 
      e2 = e1 * prat * (xnum/xdenom)
 


      if (debug) then
        write(*,*)
        write(*,*) 'e1  = ',e1 
        write(*,*) 'e2  = ',e2
      endif

      return
      end
      subroutine exact0

      implicit none

      double precision tol, xn
      double precision shockp, fprime

      integer maxnewt, niter

      logical found, debug

      include 'tube.h'

      debug = .false.

      tol = 1.0d-10

      maxnewt = 40
 
      a1 = sqrt(gamma*p1/r1)
      a4 = sqrt(gamma*p4/r4)



      found = .false.
      niter = 0

      xn =  0.5d+0 * (p1 + p4)

   10 if ( (.not. found) .and. (niter .le. maxnewt) ) then

        niter = niter + 1

        xn = xn - (shockp(xn) / fprime(xn))

        if (debug) then
          write(*,*) niter,shockp(xn),xn
        endif

        if ( abs(shockp(xn)) .lt. tol ) then
           found = .true.
        endif

        goto 10

      endif

      if (.not. found) then

         write(*,*) 'newton failed'
         write(*,*) xn,shockp(xn)
         stop

      endif

      p2 = xn 


      if (debug) then
        write(*,*)
        write(*,*) 'p1  = ',p1 
        write(*,*) 'p2  = ',p2 
        write(*,*) 'p4  = ',p4 
        write(*,*)
      endif

      return
      end
      double precision function flux(r,ru,e,eqn)
c23456789012345678901234567890123456789012345678901234567890123456789012
c
c          function flux
c
c  This function computes the flux at a face
c
c23456789012345678901234567890123456789012345678901234567890123456789012


c#######################################################################

      implicit none

      include 'comd.h'
      include 'tube.h'

      double precision r, ru, e

      integer eqn

      double precision p,u

      double precision eos


c#######################################################################

      p = eos(r,ru,e)
      u = ru/r

      if (eqn .eq. 1) then
         flux = ru
      elseif (eqn .eq. 2) then
         flux = (u * ru) + p 
      else
         flux = u * (e + p)
      endif

      return
      end
      double precision function fluxlim(fww,fw,fe,fee,vp)
c23456789012345678901234567890123456789012345678901234567890123456789012
c
c          function fluxlim
c
c  this function computes the flux limited quick face value
c
c23456789012345678901234567890123456789012345678901234567890123456789012


c#######################################################################

      implicit none

      double precision fww, fw, fe, fee, vp

      double precision fd, fc, fu

      double precision f1, f2, f3, f4, fhod, beta, flc

      double precision med, quick
 
      logical limit

c#######################################################################

      limit = .true.

      if (vp .gt. 0.0d+0) then
        fd = fe
        fc = fw
        fu = fww
      else
        fd = fw
        fc = fe
        fu = fee
      endif

      fhod = quick(fd,fc,fu)

      if (limit) then

        beta = 0.25d+0
        flc = 4.0d+0

        f1 = fc
        f2 = (beta*fc) + ( (1.0d+0-beta)*fd )
        f3 = fu + ( flc * (fc - fu) )
        f4 = med(f1,f2,f3)
        fluxlim = vp * med(f1,f4,fhod)

      else

        fluxlim = vp * fhod

      endif

      return
      end
      double precision function fluxlim2(fww,fw,fe,fee,vp)
c23456789012345678901234567890123456789012345678901234567890123456789012
c
c          function fluxlim2
c
c  this function computes the flux limited quick face value
c
c23456789012345678901234567890123456789012345678901234567890123456789012


c#######################################################################

      implicit none

      double precision fww, fw, fe, fee, vp

      double precision fd, fc, fu

      double precision f1, f2, f3, f4, fhod, beta, flc

      double precision med, quick
 
      logical limit, debug

c#######################################################################

      debug = .false.

      if (debug) then
        write(*,*)
        write(*,*) 'in fluxlim2 fee,fe,fw,fww'
        write(*,*) fee,fe,fw,fww
        write(*,*)
      endif

      limit = .true.

      if (vp .gt. 0.0d+0) then
        fd = fe
        fc = fw
        fu = fww
      else
        fd = fw
        fc = fe
        fu = fee
      endif

      fhod = quick(fd,fc,fu)

      if (limit) then

        beta = 0.25d+0
        flc = 4.0d+0

        f1 = fc
        f2 = (beta*fc) + ( (1.0d+0-beta)*fd )
        f3 = fu + ( flc * (fc - fu) )
        f4 = med(f1,f2,f3)
        fluxlim2 =  med(f1,f4,fhod)

      else

        fluxlim2 = fhod

      endif

      return
      end
      double precision function fprime(x)

      implicit none

      double precision  x, eps
      double precision  shockp

      eps = 1.0d-8

      fprime = ( shockp(x+eps) - shockp(x) ) / eps

      return
      end
      double precision function godent(uhl, uhr, hl, hr, eqn)
c23456789012345678901234567890123456789012345678901234567890123456789012
c
c          function godent
c
c  this function computes the roe/godunov face value plus entropy fix
c
c23456789012345678901234567890123456789012345678901234567890123456789012


c#######################################################################

      implicit none

      include 'comd.h'

      double precision uhl, uhr, hl, hr, ht, uht
      double precision lamr1, lamr2, laml1, laml2
      double precision deltal1, deltal2

      integer eqn

      double precision sum

      double precision flux

      integer i, j

 

c#######################################################################

      if (debug) then
        write(*,*) 'in godent eqn = ',eqn
      endif

c      do i = 1,neq
c        fr(i) = flux(uhr,hr,i)
c        fl(i) = flux(uhl,hl,i)
c      enddo

      deltau(1) = hr - hl
      deltau(2) = uhr - uhl

      call roestat(uhl, uhr, hl,hr,ht, uht)

      call eigen(ht,uht)

      do i = 1,neq 
        sum = 0.0d+0
        do j = 1,neq
          sum = sum + ( rinv(i,j) * deltau(j) )
        enddo
        alpha(i) = sum
      enddo

      deltal1 = 0.0d+0
      deltal2 = 0.0d+0

c      call eigene(hr,uhr,lamr1, lamr2)
c      call eigene(hl,uhl,laml1, laml2)
c
c  1st eigen
c
      if ( (laml1 .lt. 0.0d+0) .and.
     &     (lamr1 .gt. 0.0d+0) ) then

CVAM     deltal1 = 4.0d+0 * (lamr1 - laml1)
         deltal1 = 4.0d+0 * (lamr1 - laml1) + 1.0d-2

         if ( abs(eigval(1)) .lt. (0.5d+0 * deltal1) ) then
             eigval(1) = (  ( (eigval(1) ** 2) / deltal1 ) 
     &                    + ( 0.25d+0 * deltal1 )  )
         endif

      endif
c
c  2nd eigen
c
      if ( (laml2 .lt. 0.0d+0) .and.
     &     (lamr2 .gt. 0.0d+0) ) then

         deltal2 = 4.0d+0 * (lamr2 - laml2)

         if ( abs(eigval(2)) .lt. (0.5d+0 * deltal2) ) then
             eigval(2) = (  ( (eigval(2) ** 2) / deltal2 ) 
     &                    + ( 0.25d+0 * deltal2 )  )
         endif

      endif

      if (debug) then
         write(*,*)
         write(*,*) 'godent debug'
         write(*,*) laml1, laml2, lamr1, lamr2
         write(*,*) deltal1, deltal2
         write(*,*)
      endif

      do i = 1,neq 
        sum = 0.0d+0
        do j = 1,neq
          sum = sum - ( 0.5d+0*alpha(j)*eigvec(i,j)*abs(eigval(j)) )
        enddo
        xnumdif(i) = sum
      enddo

      do i = 1,neq
        froe(i) = ( 0.5d+0 * (fr(i) + fl(i)) ) + xnumdif(i)
      enddo

      godent = froe(eqn)

      return
      end
      double precision function godunov(uhl, uhr, hl, hr, eqn)
c23456789012345678901234567890123456789012345678901234567890123456789012
c
c          function godunov
c
c  this function computes the roe/godunov face value
c
c23456789012345678901234567890123456789012345678901234567890123456789012


c#######################################################################

      implicit none

      include 'comd.h'

      double precision uhl, uhr, hl, hr, ht, uht

      integer eqn

      double precision sum

      double precision flux

      integer i, j
 

c#######################################################################

c      do i = 1,neq
c        fr(i) = flux(uhr,hr,i)
c        fl(i) = flux(uhl,hl,i)
c      enddo

      deltau(1) = hr - hl
      deltau(2) = uhr - uhl

      call roestat(uhl, uhr, hl,hr,ht, uht)

      call eigen(ht,uht)

      do i = 1,neq 
        sum = 0.0d+0
        do j = 1,neq
          sum = sum + ( rinv(i,j) * deltau(j) )
        enddo
        alpha(i) = sum
      enddo

      do i = 1,neq 
        sum = 0.0d+0
        do j = 1,neq
          sum = sum - ( 0.5d+0*alpha(j)*eigvec(i,j)*abs(eigval(j)) )
        enddo
        xnumdif(i) = sum
      enddo

      do i = 1,neq
        froe(i) = ( 0.5d+0 * (fr(i) + fl(i)) ) + xnumdif(i)
      enddo

      godunov = froe(eqn)


      return
      end
      double precision function godunov2
     &            (rll,rl,rr,rrr,rull,rul,rur,rurr,ell,el,er,err,eqn)
c23456789012345678901234567890123456789012345678901234567890123456789012
c
c          function godunov2
c
c  this function computes the roe/godunov2 face value
c
c23456789012345678901234567890123456789012345678901234567890123456789012


c#######################################################################

      implicit none

      include 'comd.h'
      include 'tube.h'

      double precision rll,rl,rr,rrr,rull,rul,rur,rurr,ell,el,er,err

      integer eqn

      double precision rrg, rlg, rurg, rulg, erg, elg

      double precision godunov, godent, hlle


c#######################################################################

      if (gorder .eq. 1) then
        rrg  = rr
        rlg  = rl
        rurg = rur
        rulg = rul
        erg  = er
        elg  = el
      else
        call secondq(rll,rl,rr,rrr,rull,rul,rur,rurr,ell,el,er,err,
     &               rrg, rlg,rurg, rulg, erg, elg)
      endif

CVAM  if (ientro .eq. 0) then
CVAM     godunov2 = godunov(uhlg,uhrg,hlg,hrg,eqn)
CVAM  elseif(ientro .eq. 1) then
CVAM     godunov2 = godent(uhlg,uhrg,hlg,hrg,eqn)
CVAM  else
         godunov2 = hlle(rrg,rlg,rurg,rulg,erg,elg,eqn)
CVAM  endif


      return
      end
      double precision function hlle(rrg,rlg,rurg,rulg,erg,elg,eqn)
c23456789012345678901234567890123456789012345678901234567890123456789012
c
c          function hlle
c
c  this function computes the roe/hlle face value
c
c23456789012345678901234567890123456789012345678901234567890123456789012


c#######################################################################

      implicit none

      include 'comd.h'
      include 'tube.h'

      double precision rrg,rlg,rurg,rulg,erg,elg
      integer eqn

      double precision laml1, laml2, laml3
      double precision lamr1, lamr2, lamr3
      double precision sl, sr


      double precision flux

      integer i, j, ispeed
 

c#######################################################################

      ispeed = 1

      do i = 1,neq
        fr(i) = flux(rrg,rurg,erg,i)
        fl(i) = flux(rlg,rulg,elg,i)
      enddo

      deltau(1) = rrg  - rlg
      deltau(2) = rurg - rulg
      deltau(3) = erg  - elg

CVAM  call roestat(uhl,uhr,hl,hr,ht,uht)

CVAM  call eigene(ht,uht,lamt1, lamt2)
      call eigene(rrg,rurg,erg,lamr1,lamr2,lamr3)
      call eigene(rlg,rulg,elg,laml1,laml2,laml3)

CVAM  if (ispeed .eq. 1) then
CVAM    sl = min(laml1,lamt1)
CVAM    sr = max(lamt2,lamr2)
CVAM  else
        sl = min(laml1,lamr1)
        sr = max(laml3,lamr3)
CVAM  endif


      do i = 1,neq
        froe(i) = ( (sr*fl(i)) - (sl*fr(i)) + (sl*sr*deltau(i)) )
     &            / (sr-sl)
      enddo

      hlle = froe(eqn)


      return
      end
      double precision function med(x1,x2,x3)
c23456789012345678901234567890123456789012345678901234567890123456789012
c
c          function med
c
c  this function computes the median of three numbers
c
c23456789012345678901234567890123456789012345678901234567890123456789012


c#######################################################################

      implicit none

      double precision x1, x2, x3
      double precision xhi, xlo

c#######################################################################

      xhi = max(x1,x2,x3)
      xlo = min(x1,x2,x3)

      med = x1 + x2 + x3 - xhi - xlo

      return
      end
      double precision function mom
     &            (rhoee,  rhoe,  rhop,  rhow,  rhoww,
     &             rhouee, rhoue, rhoup, rhouw, rhouww,
     &             ergee,  erge,  ergp,  ergw,  ergww,
     &                                      jmom,xold)
c
c  This function computes the residual
c  for the 1-D momentum equation
c
c
      implicit none

      include 'comd.h'
      include 'tube.h'
c
c     input variables
c
      double precision rhoee,  rhoe,  rhop,  rhow,  rhoww
      double precision rhouee, rhoue, rhoup, rhouw, rhouww
      double precision ergee,  erge,  ergp,  ergw,  ergww
      double precision xold(mx*neq)
c
      integer jmom
c
c     local variables
c
      double precision theta1
      integer jru
c
c  new
c
      double precision velfw, velfe
      double precision vele,velp,velw
      double precision fluxe, fluxw
      double precision uurhoe, uurhow
      double precision pressee, presse, pressp,pressw, pressww
      double precision rupee, rupe, rupp, rupw, rupww
      double precision uee, ue, up, uw, uww
      double precision source
c
c old
c
      double precision rhooee,  rhooe,  rhoop,  rhoow,  rhooww
      double precision rhouoee, rhouoe, rhouop, rhouow, rhouoww
      double precision ergoee,  ergoe,  ergop,  ergow,  ergoww
      double precision velfow, velfoe
      double precision veloe,velop,velow
      double precision fluxoe, fluxow
      double precision uurhooe, uurhoow
      double precision pressoee, pressoe, pressop, pressow, pressoww
      double precision rupoee, rupoe, rupop, rupow, rupoww
      double precision uoee, uoe, uop, uow, uoww
      double precision sourceo

      double precision eps
c
c functions
c
      double precision godunov2, eos
      double precision upwind, fluxlim
c
c
c ******************************************************************
c
c
      eps = 1.0d-32
c
      jru = (neq*jmom) - 1

c########################
c
c      NEW
c
c########################

      call Setpbcn(rhoee,  rhoe,  rhop,  rhow,  rhoww,
     &             rhouee, rhoue, rhoup, rhouw, rhouww,
     &             ergee,  erge,  ergp,  ergw,  ergww,
     &                     vele,  velp,  velw,         jmom)

      presse  = eos(rhoe, rhoue, erge )
      pressw  = eos(rhow, rhouw, ergw )

      velfe = 0.5d+0 * (vele + velp)
      velfw = 0.5d+0 * (velw + velp)

      if (ihod .eq. 1) then

        uurhoe = upwind(rhoup,rhoue,velfe)
        uurhow = upwind(rhouw,rhoup,velfw)

      elseif (ihod .eq. 2) then

        uurhoe = fluxlim(rhouw,rhoup,rhoue,rhouee,velfe)
        uurhow = fluxlim(rhouww,rhouw,rhoup,rhoue,velfw)

      endif

      if (ihod .eq. 3) then
        fluxe = (dt/dx) * godunov2(rhow, rhop, rhoe, rhoee,
     &                             rhouw,rhoup,rhoue,rhouee,
     &                             ergw, ergp, erge, ergee,2)
        fluxw = (dt/dx) * godunov2(rhoww, rhow, rhop, rhoe,
     &                             rhouww,rhouw,rhoup,rhoue,
     &                             ergww, ergw, ergp, erge,2)
      else
        fluxe = (dt/dx) * ( uurhoe + (0.5d+0 * presse) )
        fluxw = (dt/dx) * ( uurhow + (0.5d+0 * pressw) )
      endif


      source = 0.0d+0

c########################
c
c      OLD
c
c########################

      call Setpbc(jmom,xold,
     &             rhooee,  rhooe,  rhoop,  rhoow,  rhooww,
     &             rhouoee, rhouoe, rhouop, rhouow, rhouoww,
     &             ergoee,  ergoe,  ergop,  ergow,  ergoww,
     &                      veloe,  velop,  velow)

      call Setpbcn(rhooee,  rhooe,  rhoop,  rhoow,  rhooww,
     &             rhouoee, rhouoe, rhouop, rhouow, rhouoww,
     &             ergoee,  ergoe,  ergop,  ergow,  ergoww,
     &                      veloe,  velop,  velow,         jmom)

      pressoe  = eos(rhooe, rhouoe, ergoe)
      pressow  = eos(rhoow, rhouow, ergow)

      velfoe = 0.5d+0 * (veloe + velop)
      velfow = 0.5d+0 * (velow + velop)

      if (ihod .eq. 1) then

        uurhooe = upwind(rhouop,rhouoe,velfoe)
        uurhoow = upwind(rhouow,rhouop,velfow)

      elseif (ihod .eq. 2) then

        uurhooe = fluxlim(rhouow,rhouop,rhouoe,rhouoee,velfoe)
        uurhoow = fluxlim(rhouoww,rhouow,rhouop,rhouoe,velfow)

      endif

      if (ihod .eq. 3) then
        fluxoe = (dt/dx) * godunov2(rhoow, rhoop, rhooe, rhooee,
     &                              rhouow,rhouop,rhouoe,rhouoee,
     &                              ergow, ergop, ergoe, ergoee,2)
        fluxow = (dt/dx) * godunov2(rhooww, rhoow, rhoop, rhooe,
     &                              rhouoww,rhouow,rhouop,rhouoe,
     &                              ergoww, ergow, ergop, ergoe,2)
      else
        fluxoe = (dt/dx) * ( uurhooe + (0.5d+0 * pressoe) )
        fluxow = (dt/dx) * ( uurhoow + (0.5d+0 * pressow) )
      endif

      sourceo = 0.0d+0


c########################
c
c      FUNCTION
c
c########################

      theta1 = 1.0d+0 - theta
      mom =  (rhoup - xold(jru)) 
     &    + (  theta  * ( (fluxe  - fluxw ) - source  )  )
     &    + (  theta1 * ( (fluxoe - fluxow) - sourceo )  )
CVAM
      if (probnum .eq. 3) then
        mom = 0.0d+0
      endif
CVAM
      if (debug) then
        write(*,*)
        write(*,*) 'mom(',jmom,') = ',mom,' theta = ',theta
        write(*,*) 'fluxe = ',fluxe,' fluxw = ',fluxw
        write(*,*) 'fluxoe = ',fluxoe,' fluxow = ',fluxow
        write(*,*) 'presse = ',presse,'pressw = ',pressw
        write(*,*) 'pressoe = ',pressoe,'pressow = ',pressow
        write(*,*)
      endif

      return
      end
      double precision function quick(fd, fc, fu)
c23456789012345678901234567890123456789012345678901234567890123456789012
c
c          function quick
c
c  this function computes the quick face value
c
c23456789012345678901234567890123456789012345678901234567890123456789012


c#######################################################################

      implicit none

      double precision fd, fc, fu

c#######################################################################

      quick = ( (3.0d+0 * fd) + (6.0d+0 * fc) - fu ) / 8.0d+0

      return
      end
      double precision function  rexact(x,t)

      implicit none

      double precision x,t
      double precision xot, head, tail, contact, ufan
      double precision xpow, grat, urat
      double precision uexact


      logical debug

      include 'tube.h'

      debug = .false.


      if (t .le. 0.0d+0) then
        if (x .gt. 0.0d+0) then
          rexact = r1
        else
          rexact = r4
        endif
      else

       xot = x/t
       head = -a4
       tail = v3 - a3
       contact = v2

       if (xot .lt. head) then
          rexact = r4
       elseif (xot .gt. sspd) then
          rexact = r1
       elseif (xot .gt. contact) then
          rexact = r2
       elseif (xot .gt. tail) then
          rexact = r3
       else
          ufan = uexact(x,t)
          grat = (gamma - 1.0d+0) / 2.0d+0
          xpow = 1.0d+0 / grat
          urat = ufan / a4
          rexact = r4 * (  ( 1.0d+0 - (grat * urat) ) ** xpow  )
       endif

      endif


      if (debug) then
        write(*,*)
        write(*,*) 'rexact(',x,',',t,') = ',rexact
        write(*,*)
      endif

      return
      end
      subroutine roestat(uhl, uhr, hl,hr,ht,uht)
c23456789012345678901234567890123456789012345678901234567890123456789012
c
c          subroutine roestat
c
c  This subroutine computes the roe state at a face
c
c23456789012345678901234567890123456789012345678901234567890123456789012


c#######################################################################

      implicit none

      include 'comd.h'

      double precision uhl, uhr, hl, hr, ht, uht

      double precision ul, ur, shl, shr, xnum, xdenom
 


c#######################################################################

      ul = uhl / hl
      ur = uhr / hr

      shl = sqrt(hl)
      shr = sqrt(hr)

      xnum = (shl * ul) + (shr * ur)
      xdenom = shl + shr

      ht  = 0.5d+0 * (hl + hr)
      uht = ht * ( xnum / xdenom )

      return
      end
      subroutine rval2

      implicit none

      double precision prat, grat, xnum, xdenom


      logical debug

      include 'tube.h'

      debug = .false.

      prat = p2/p1
      grat = (gamma + 1.0d+0) / (gamma - 1.0d+0)

      xnum = 1.0d+0 + (grat * prat)
      xdenom = grat + prat
 
      r2 = r1 * (xnum/xdenom)
 


      if (debug) then
        write(*,*)
        write(*,*) 'r1  = ',r1 
        write(*,*) 'r2  = ',r2
      endif

      return
      end
      subroutine  secondq
     &      (rll,rl,rr,rrr,rull,rul,rur,rurr,ell,el,er,err,
     &               rrg, rlg,rurg, rulg, erg, elg)
c23456789012345678901234567890123456789012345678901234567890123456789012
c
c          subroutine secondq
c
c  this subroutine computes the second order (based on quick) left
c  and right states for the godunov solver.
c
c23456789012345678901234567890123456789012345678901234567890123456789012


c#######################################################################

      implicit none

      include 'comd.h'

      double precision rll,rl,rr,rrr,rull,rul,rur,rurr,ell,el,er,err
      double precision rrg, rlg,rurg, rulg, erg, elg



      double precision veld, ull,ul,ur,urr, ulg, urg

      double precision fluxlim2


c#######################################################################

c
c  compute the velocities
c
      ull = rull/rll
      ul  = rul /rl 
      ur  = rur /rr
      urr = rurr/rrr

c
c  compute the left state first
c
      veld = 1.0d+0

      rlg = fluxlim2(rll,rl,rr,rrr,veld)
      ulg = fluxlim2(ull,ul,ur,urr,veld)
      rulg = rlg * ulg
      elg = fluxlim2(ell,el,er,err,veld)
c
c  now compute the right state
c
      veld = -1.0d+0

      rrg = fluxlim2(rll,rl,rr,rrr,veld)
      urg = fluxlim2(ull,ul,ur,urr,veld)
      rurg = rrg * urg
      erg = fluxlim2(ell,el,er,err,veld)



      return
      end
      double precision function shockp(x)

      implicit none

      double precision x
      double precision xnum, xdenom, xpow, prat, prat2, prat4, gm, gp
      logical debug

      include 'tube.h'

      debug = .false.


      if (debug) then
         write(*,*)
         write(*,*) 'gamma = ',gamma
         write(*,*) 'a1 = ',a1
         write(*,*) 'a4 = ',a4
         write(*,*) 'p1 = ',p1
         write(*,*) 'p2 = ',x
         write(*,*)
      endif

      xnum = (gamma - 1.0d+0) * (a1/a4) * ( (x/p1) - 1.0d+0 )
      xdenom = sqrt  (  2.0d+0 * gamma * ( (2.0d+0*gamma)
     &            + (gamma + 1.0d+0) * ((x/p1) - 1) )  )
      xpow = (-2.0d+0 * gamma) / (gamma - 1.0d+0)

      shockp = (x/p1)*((1.0d+0-(xnum/xdenom))**xpow) - (p4/p1)


      if (debug) then
         write(*,*)
         write(*,*) 'xnum = ',xnum
         write(*,*) 'gamma = ',gamma
         write(*,*) 'a1 = ',a1
         write(*,*) 'a4 = ',a4
         write(*,*) 'p1 = ',p1
         write(*,*) 'xdenom = ',xdenom
         write(*,*) 'xpow = ',xpow
         write(*,*) 'shockp = ',shockp
         write(*,*) 'p2 = ',x
         write(*,*)
      endif

      return
      end
      double precision function  uexact(x,t)

      implicit none

      double precision x,t
      double precision xot, head, tail


      logical debug

      include 'tube.h'

      debug = .false.

      if (debug) then
        write(*,*) 
        write(*,*) 't = ',t
        write(*,*) 'x = ',x
        write(*,*) 'a4 = ',a4
        write(*,*) 'v3 = ',v3
        write(*,*) 'a3 = ',a3
        write(*,*) 
      endif

      if (t .le. 0.0d+0) then
        uexact = 0.0d+0
      else

       xot = x/t
       head = -a4
       tail = v3 - a3

       if (xot .lt. head) then
          uexact = 0.0d+0
       elseif (xot .gt. sspd) then
          uexact = 0.0d+0
       elseif (xot .gt. tail) then
          uexact = v2
       else
          uexact = (2.0d+0 / (gamma + 1.0d+0))* (a4 + xot)
       endif

      endif


      if (debug) then
        write(*,*)
CVAM    write(*,*) 'x = ',x,' t = ',t
        write(*,*) 'uexact = ',uexact
        write(*,*)
      endif

      return
      end
      double precision function upwind(fw, fe, vp)
c23456789012345678901234567890123456789012345678901234567890123456789012
c
c          function upwind
c
c  this function computes the upwind face value
c
c23456789012345678901234567890123456789012345678901234567890123456789012


c#######################################################################

      implicit none

      double precision fw, fe, vp

c#######################################################################

      if (vp .gt. 0.0) then
         upwind = vp * fw
      else
         upwind = vp * fe
      endif

      return
      end
      subroutine uval2

      implicit none

      double precision prat, grat1, grat2, arat, xnum


      logical debug

      include 'tube.h'

      debug = .false.

      prat = p2/p1
      grat1 = (gamma - 1.0d+0) / (gamma + 1.0d+0)
      grat2 = (2.0d+0 * gamma) / (gamma + 1.0d+0)
      arat = a1/gamma

      xnum = sqrt ( grat2 / (prat + grat1) )

      v2 = arat * (prat - 1.0d+0) * xnum

      if (debug) then
        write(*,*)
        write(*,*) 'v2  = ',v2
      endif

      return
      end
      subroutine val3

      implicit none

      double precision prat, rpow, epow, p3t


      logical debug

      include 'tube.h'

      debug = .false.


      p3 = p2

      prat = p3/p4

      rpow = 1.0d+0 / gamma

      r3 = r4 * ( prat ** rpow )

      epow = (gamma - 1.0d+0) / gamma

      e3 = e4 * ( (p3/p4) ** epow )

      p3t = (gamma - 1.0d+0) * r3 * e3

      a3 = sqrt(gamma*p3/r3)

      if (debug) then
        write(*,*)
        write(*,*) 'a3 = ',a3
        write(*,*) 'r3 = ',r3
        write(*,*) 'e3 = ',e3
        write(*,*) 'p3 = ',p3
        write(*,*) 'p3t = ',p3t,' error = ',p3-p3t
        write(*,*)
      endif

      return
      end
      subroutine wval

      implicit none

      double precision prat, grat, xnum


      logical debug

      include 'tube.h'

      debug = .false.

      prat = p2/p1
      grat = (gamma + 1.0d+0) / (2.0d+0 * gamma)

      xnum = ( grat * (prat - 1.0d+0) ) + 1.0d+0
 
      sspd = a1 * sqrt(xnum)
 


      if (debug) then
        write(*,*)
        write(*,*) 'sspd  = ',sspd
      endif

      return
      end
