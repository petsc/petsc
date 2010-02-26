#include "finclude/petscdef.h"

!  Error handler that aborts when error is detected
!
      subroutine HE1(ierr,line)
      use mex35f90
      use mex35f90interfaces

      call PetscError(ierr,line,0,'',ierr)
      call MPI_Abort(PETSC_COMM_WORLD,ierr,ierr)
      return
      end      
#define CHKQ(n) if(n .ne. 0)call HE1(n,__LINE__)

!  Error handler forces traceback of where error occurred
!
      subroutine HE2(ierr,line)
      use mex35f90
      use mex35f90interfaces

      call PetscError(ierr,line,0,'',ierr)
      return
      end      
#define CHKR(n) if(n .ne. 0)then;call HE2(n,__LINE__);return;endif

!
!      Implements a nonlinear solver for a simple domain
!     consisting of 2 zero dimensional problems linked by
!     2 one dimensional problems.
!
!                           Channel1
!                       -------------------------
!               Pool 1                              Pool 2
!                       -------------------------
!                           Channel2
!VAM
!VAM
!VAM
!VAM                         Hot Pool 1
!VAM                 |                       |
!VAM                 |                       |
!VAM                 |                       |
!VAM                 |                       |
!VAM  Core Channel 2 |                       | IHX Channel 1
!VAM                 |                       |
!VAM                 |                       |
!VAM                 |                       |
!VAM                 |                       |
!VAM                         Cold Pool 2
!VAM
!
!     For Newton's method with block Jacobi (one block for
!     each channel and one block for each pool) run with
!
!       -dmmg_iscoloring_type global -snes_mf_operator -pc_type lu
!       -help lists all options

      program ex35f90
      use mex35f90
!     use mex35f90interfaces
#include "finclude/petscsys.h"
#include "finclude/petscviewer.h"
#include "finclude/petscvec.h"


      DMMGArray        dmmg
      DMMG             dmmglevel 
      PetscErrorCode   ierr
      DA               da
      DMComposite      dm
      type(appctx)     app
      external         FormInitialGuess,FormFunction,FormGraph
      Vec              x

      double precision time
      integer i
!BARRY
       PetscViewer        view0,view1
!BARRY

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

!      Create the composite DM object that manages the grid

      call DMCompositeCreate(PETSC_COMM_WORLD,dm,ierr);CHKR(ierr)
      call DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,app%nxc,app%nc,1,PETSC_NULL_INTEGER,da,ierr)
      CHKR(ierr)
      call DMCompositeAddDM(dm,da,ierr);CHKR(ierr)
      call DMCompositeAddArray(dm,0,app%np,ierr);CHKR(ierr)
      call DMCompositeAddDM(dm,da,ierr);CHKR(ierr)
      call DADestroy(da,ierr);CHKR(ierr)
      call DMCompositeAddArray(dm,0,app%np,ierr);CHKR(ierr)

!       Create solver object and associate it with the unknowns (on the grid)

      call DMMGCreate(PETSC_COMM_WORLD,1,PETSC_NULL_OBJECT,dmmg,ierr);CHKR(ierr)
      call DMMGSetDM(dmmg,dm,ierr);CHKR(ierr)
      call DMCompositeDestroy(dm,ierr);CHKR(ierr)
      call DMMGSetUser(dmmg,0,app,ierr);CHKR(ierr)  ! currently only one level solver
      call DMMGSetInitialGuess(dmmg,FormInitialGuess,ierr);CHKR(ierr)
      call DMMGSetSNES(dmmg,FormFunction,PETSC_NULL_OBJECT,ierr);CHKR(ierr)
      call DMMGSetFromOptions(dmmg,ierr)
!BARRY
      call PetscViewerDrawOpen(PETSC_COMM_WORLD,PETSC_NULL_CHARACTER,'core',0,0,300,300,view0,ierr)
      CHKR(ierr)
      call PetscViewerDrawOpen(PETSC_COMM_WORLD,PETSC_NULL_CHARACTER,'ihx',320,0,300,300,view1,ierr)
      CHKR(ierr)
!BARRY

      call DMMGGetX(dmmg,x,ierr);CHKR(ierr)
      call VecDuplicate(x,app%xold,ierr);CHKR(ierr)
      call VecCopy(x,app%xold,ierr);CHKR(ierr)
!BARRY
       write(*,*) 'Before call to FormGraph'
       call DMMGArrayGetDMMG(dmmg,dmmglevel,ierr);CHKR(ierr)
       call FormGraph(dmmglevel,x,view0,view1,ierr);CHKR(ierr)
!BARRY

      time = 0.0d+0

      write(*,*)
      write(*,*)  'initial time'
      write(*,*)

      do i = 1,app%nstep

        time = time + app%dt

        write(*,*)
        write(*,*)  'time =',time
        write(*,*)
!
!  copy new to old
!
!
!   Solve the nonlinear system
!
        call DMMGSolve(dmmg,ierr);CHKR(ierr)
        call DMMGSetInitialGuess(dmmg,PETSC_NULL_FUNCTION,ierr);CHKR(ierr)

        call DMMGGetX(dmmg,x,ierr);CHKR(ierr)
        call VecCopy(x,app%xold,ierr);CHKR(ierr)
!BARRY
       call DMMGArrayGetDMMG(dmmg,dmmglevel,ierr);CHKR(ierr)
       call FormGraph(dmmglevel,x,view0,view1,ierr);CHKR(ierr)
!BARRY
      enddo
!BARRY
      call PetscViewerDestroy(view0,ierr);CHKR(ierr)
      call PetscViewerDestroy(view1,ierr);CHKR(ierr)
!BARRY
      write(*,*)
      write(*,*)  'final time'
      write(*,*)

      call VecDestroy(app%xold,ierr);CHKR(ierr)
      call DMMGDestroy(dmmg,ierr);CHKR(ierr)
      call PetscFinalize(ierr)
      end

      subroutine FormFunction(snes,x,f,dmmg,ierr)
      use mex35f90
      use mex35f90interfaces

      SNES snes
      Vec x,f
      DMMG dmmg
      PetscErrorCode ierr

      DMComposite dm
      Vec  fvc1,fvc2,xvc1,xvc2
      PetscInt np,i
      DA da
      type(DALocalInfof90) dainfo
      type(poolfield), pointer :: fHotPool,fColdPool
      type(poolfield), pointer :: xHotPool,xColdPool
      type(channelfield), pointer :: fIHX(:),fCore(:),xIHX(:),xCore(:)
      type(appctx), pointer :: app
      PetscMPIInt rank

      double precision dhhpl, mult, dhcpl, phpco, pcpio, pcpci, phpii

      logical debug

      debug = .false.

      call DMMGGetUser(dmmg,app,ierr);CHKR(ierr)   ! access user context

!         Access the grid information

      call DMMGGetDM(dmmg,dm,ierr);CHKR(ierr)
      call DMCompositeGetEntries(dm,da,np,da,np,ierr);CHKR(ierr) ! get the da's and sizes that define the unknowns
      call DAGetLocalInfof90(da,dainfo,ierr);CHKR(ierr)

!        Access the local (ghosted) parts of x

      call DMCompositeGetLocalVectors(dm,xvc1,xHotPool,xvc2,xColdPool,ierr);CHKR(ierr)
      call DMCompositeScatter(dm,x,xvc1,xHotPool,xvc2,xColdPool,ierr);CHKR(ierr)

      call DAVecGetArrayf90(da,xvc1,xIHX,ierr);CHKR(ierr)
      call DAVecGetArrayf90(da,xvc2,xCore,ierr);CHKR(ierr)

!       Access the global (non-ghosted) parts of f

      call DMCompositeGetAccess(dm,f,fvc1,fHotPool,fvc2,fColdPool,ierr);CHKR(ierr)
      call DAVecGetArrayf90(da,fvc1,fIHX,ierr);CHKR(ierr)
      call DAVecGetArrayf90(da,fvc2,fCore,ierr);CHKR(ierr)

!BARRY
!
!  At this point I need to create old time values from "xold" passed in through
!  app for
!
!  xHotPool%vol, xHotPool%vel, xColdPool%vol, xColdPool%vel
!  xIHX(i)%press, xCore(i)%press
!
!  I don't know how to do this.
!
!BARRY

      call MPI_Comm_rank(app%comm,rank,ierr);CHKR(ierr)
!      fPool only lives on zeroth process, ghosted xPool lives on all processes
      if (rank == 0) then
!
!  compute velocity into hotpool from core
!
         dhhpl = app%dhhpl0 + ( (xHotPool%vol - app%hpvol0) / app%ahp )
         phpco = app%P0 + ( app%rho * app%grav * (dhhpl - app%dhco) )
         mult = app%dt / (app%dxc * app%rho)
         fHotPool%vel = xHotPool%vel - (mult * (xCore(app%nxc-1)%press - phpco) ) + (abs(xHotPool%vel) * xHotPool%vel)
!
! compute change in hot pool volume
!
         fHotPool%vol = xHotPool%vol - app%hpvol0 - (app%dt * app%acore * (xHotPool%vel -xColdPool%vel) )
!
!  compute velocity into coldpool from IHX
!
         dhcpl = app%dhcpl0 + ( (xColdPool%vol - app%cpvol0) / app%acp )
         pcpio = app%P0 + ( app%rho * app%grav * (dhcpl - app%dhio) )
         mult = app%dt / (app%dxc * app%rho)
         fColdPool%vel = xColdPool%vel - (mult * (xIHX(app%nxc-1)%press - pcpio) ) + (abs(xColdPool%vel) * xColdPool%vel)
!
!  compute the change in cold pool volume
!
         fColdPool%vol = xColdPool%vol - app%cpvol0 - (app%dt * app%aihx * (xColdPool%vel - xHotPool%vel) )
      endif
!
!  Compute the pressure distribution in IHX and core
!
      pcpci = app%P0 + ( app%rho * app%grav * (dhcpl - app%dhci) )
      phpii = app%P0 + ( app%rho * app%grav * (dhhpl - app%dhii) )

      do i=dainfo%xs,dainfo%xs+dainfo%xm-1

        fIHX(i)%press = xIHX(i)%press - phpii - (app%rho * app%grav * dble(i) * app%dxi)

        fCore(i)%press = xCore(i)%press - pcpci + (app%rho * app%grav * dble(i) * app%dxc)

      enddo

      if (debug) then
        write(*,*)
        write(*,*) 'Hot vel,vol ',xHotPool%vel,xHotPool%vol
        write(*,*) 'delta p = ', xCore(app%nxc-1)%press - phpco,xCore(app%nxc-1)%press,phpco
        write(*,*)

        do i=dainfo%xs,dainfo%xs+dainfo%xm-1
          write(*,*) 'xIHX(',i,')%press = ',xIHX(i)%press
        enddo

        write(*,*)
        write(*,*) 'Cold vel,vol ',xColdPool%vel,xColdPool%vol
        write(*,*) 'delta p = ',xIHX(app%nxc-1)%press - pcpio,xIHX(app%nxc-1)%press, pcpio
        write(*,*)


        do i=dainfo%xs,dainfo%xs+dainfo%xm-1
          write(*,*) 'xCore(',i,')%press = ',xCore(i)%press
        enddo

      endif

      call DAVecRestoreArrayf90(da,xvc1,xIHX,ierr);CHKR(ierr)
      call DAVecRestoreArrayf90(da,xvc2,xCore,ierr);CHKR(ierr)
      call DMCompositeRestoreLocalVectors(dm,xvc1,xHotPool,xvc2,xColdPool,ierr);CHKR(ierr)

      call DAVecRestoreArrayf90(da,fvc1,fIHX,ierr);CHKR(ierr)
      call DAVecRestoreArrayf90(da,fvc2,fCore,ierr);CHKR(ierr)
      call DMCompositeRestoreAccess(dm,f,fvc1,fHotPool,fvc2,fColdPool,ierr);CHKR(ierr)

      return
      end
      subroutine FormGraph(dmmg,x,view0,view1,ierr)
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
      use mex35f90
      use mex35f90interfaces

      Vec       x,xvc1,xvc2   !,corep,ihxp
      DMMG      dmmg
      PetscErrorCode ierr
      DMComposite dm
      PetscInt np            !,i
      DA da
      type(DALocalInfof90) dainfo
      type(poolfield), pointer :: HotPool,ColdPool
      type(poolfield), pointer :: xHotPool,xColdPool
      type(channelfield), pointer :: xIHX(:),xCore(:)
      type(appctx), pointer :: app
      PetscMPIInt rank

      PetscViewer        view0,view1

      integer iplotnum
      save iplotnum
      character*8 grfile

      data iplotnum / -1 /
!BARRY
!
!  This is my attempt to get the information out of vector "x" to plot
!  it.  I need to be able to build  xCore(i)%press and xIHX(i)%press
!  from the vector "x" and I do not know how to do this
!
!BARRY

      write(*,*) 
      write(*,*) 'inside of FormGraph'
      write(*,*) 

      call DMMGGetUser(dmmg,app,ierr);CHKR(ierr)   ! access user context

      write(*,*) 
      write(*,*) 'after DMMGGetUser'
      write(*,*) 

!         Access the grid information

      call DMMGGetDM(dmmg,dm,ierr);CHKR(ierr)

      write(*,*) 
      write(*,*) 'after DMMGGetDM'
      write(*,*) 

      call DMCompositeGetEntries(dm,da,np,da,np,ierr);CHKR(ierr) ! get the da's and sizes that define the unknowns
      write(*,*) 
      write(*,*) 'after DMCompositeGetEntries'
      write(*,*) 
      call DAGetLocalInfof90(da,dainfo,ierr);CHKR(ierr)
      write(*,*) 
      write(*,*) 'after DAGetLocalInfof90'
      write(*,*) 
!BARRY
!
! I think that the code dies in the call below.
!
!BARRY
      call DMCompositeGetLocalVectors(dm,xvc1,xHotPool,xvc2,xColdPool,ierr);CHKR(ierr)
      call DMCompositeScatter(dm,x,xvc1,xHotPool,xvc2,xColdPool,ierr);CHKR(ierr)
      call DAVecGetArrayf90(da,xvc1,xIHX,ierr);CHKR(ierr)
      write(*,*) 
      write(*,*) 'after DAVecGetArrayf90(da,xvc1,xIHX,ierr)'
      write(*,*) 
      call DAVecGetArrayf90(da,xvc2,xCore,ierr);CHKR(ierr)
      write(*,*) 
      write(*,*) 'after DAVecGetArrayf90(da,xvc2,xCore,ierr)'
      write(*,*) 


!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
      iplotnum = iplotnum + 1
      ierr   = 0
!
!
!  plot corep vector
!
      call VecView(xvc1,view0,ierr);CHKR(ierr)
!
!  make xmgr plot of corep
!
      write(grfile,4000) iplotnum
 4000 format('CoreP',i3.3)

      open (unit=44,file=grfile,status='unknown')

      do i = 1,app%nxc
        write(44,1000) dble(i)*app%dxc, xCore(i)%press
      enddo

      close(44)
!
!  plot ihxp vector
!
      call VecView(xvc2,view1,ierr);CHKR(ierr)
!
!  make xmgr plot of ihxp
!
      write(grfile,8000) iplotnum
 8000 format('IHXPr',i3.3)

      open (unit=44,file=grfile,status='unknown')

      do i = 1,app%nxc
        write(44,1000) dble(i)*app%dxi, xIHX(i)%press
      enddo

      close(44)



 1000 format(3(e18.12,2x))

      call DAVecRestoreArrayf90(da,xvc1,xIHX,ierr);CHKR(ierr)
      call DAVecRestoreArrayf90(da,xvc2,xCore,ierr);CHKR(ierr)
      call DMCompositeRestoreLocalVectors(dm,xvc1,xHotPool,xvc2,xColdPool,ierr);CHKR(ierr)
      return
      end


      subroutine FormInitialGuess(dmmg,v,ierr)
      use mex35f90
      use mex35f90interfaces

      DMMG dmmg
      Vec v
      PetscErrorCode ierr

      DMComposite dm
      Vec  vc1,vc2
      PetscInt np,i
      DA da
      type(DALocalInfof90) dainfo
      type(poolfield), pointer :: HotPool,ColdPool
      type(channelfield), pointer :: IHX(:),Core(:)
      type(appctx), pointer :: app
      PetscMPIInt rank

      double precision pcpci, phpii

      logical debug

      debug = .false.

      call DMMGGetUser(dmmg,app,ierr);CHKR(ierr)   ! access user context

!       Access the grid information

      call DMMGGetDM(dmmg,dm,ierr);CHKR(ierr)
      call DMCompositeGetEntries(dm,da,np,da,np,ierr);CHKR(ierr) ! get the da's and sizes that define the unknowns
      call DAGetLocalInfof90(da,dainfo,ierr);CHKR(ierr)

!      Acess the vector unknowns in global (nonghosted) form

      call DMCompositeGetAccess(dm,v,vc1,HotPool,vc2,ColdPool,ierr);CHKR(ierr)
      call DAVecGetArrayf90(da,vc1,IHX,ierr);CHKR(ierr)
      call DAVecGetArrayf90(da,vc2,Core,ierr);CHKR(ierr)

      call MPI_Comm_rank(app%comm,rank,ierr);CHKR(ierr)
!
!  Set the input values
!

       app%P0 = 1.0d+5
       app%rho = 1.0d+3
       app%grav = 9.8d+0

       app%dhhpl0 = 12.2d+0
       app%dhcpl0 = 10.16d+0

       app%lenc = 3.0d+0
       app%leni = 4.46d+0

       app%dhci = 0.83d+0
       app%dhco = app%dhci + app%lenc

       app%dhii = 7.83d+0
       app%dhio = app%dhii - app%leni

       app%dxc = app%lenc / dble(app%nxc)
       app%dxi = app%leni / dble(app%nxc)

       app%dt = 1.0d+0

       app%ahp = 7.0d+0
       app%acp = 7.0d+0

       app%acore = 0.8d+0
       app%aihx  = 5.0d-2

       app%hpvol0 = app%dhhpl0 * app%ahp
       app%cpvol0 = app%dhcpl0 * app%acp
!
!      the pools are only unknowns on process 0
!
      if (rank == 0) then
         HotPool%vel = -1.0d+0
         HotPool%vol = app%hpvol0
         ColdPool%vel = 1.0d+0
         ColdPool%vol = app%cpvol0
      endif
!
!  Construct and initial guess at the pressure
!
      pcpci = app%P0 + ( app%rho * app%grav * (app%dhcpl0 - app%dhci) )
      phpii = app%P0 + ( app%rho * app%grav * (app%dhhpl0 - app%dhii) )

      if (debug) then
        write(*,*)
        write(*,*) 'pcpci = ',pcpci
        write(*,*) 'phpii = ',phpii
        write(*,*) 'app%P0 = ',app%P0
        write(*,*) 'dhcpl0 - app%dhci ',app%dhcpl0 - app%dhci,app%dhcpl0, app%dhci
        write(*,*) 'dhhpl0 - app%dhii ',app%dhhpl0 - app%dhii,app%dhhpl0, app%dhii
        write(*,*)
      endif

      do i=dainfo%xs,dainfo%xs+dainfo%xm-1

        IHX(i)%press = phpii  + (app%rho * app%grav * dble(i) * app%dxi)

        Core(i)%press = pcpci - (app%rho * app%grav * dble(i) * app%dxc)

      enddo

      call DAVecRestoreArrayf90(da,vc1,IHX,ierr);CHKR(ierr)
      call DAVecRestoreArrayf90(da,vc2,Core,ierr);CHKR(ierr)
      call DMCompositeRestoreAccess(dm,v,vc1,HotPool,vc2,ColdPool,ierr);CHKR(ierr)

      CHKMEMQ
      return
      end
