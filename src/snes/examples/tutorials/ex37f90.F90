#include "finclude/petscdef.h"
!
!   Notes:
!     This uses Fortran 90 free-form, this means the lines can be up to 132 columns wide
!       CHKR(ierr) is put on the same line as the call except when it violates the 132 columns
!
!     Eventually this file can be split into several files
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Error handler that aborts when error is detected
!
      subroutine HE1(ierr,line)
      use mex37f90
      use mex37f90interfaces

      call PetscError(ierr,line,0,'',ierr)
      call MPI_Abort(PETSC_COMM_WORLD,ierr,ierr)
      return
      end      
#define CHKQ(n) if(n .ne. 0)call HE1(n,__LINE__)

!  Error handler forces traceback of where error occurred
!
      subroutine HE2(ierr,line)
      use mex37f90
      use mex37f90interfaces

      call PetscError(ierr,line,0,'',ierr)
      return
      end      
#define CHKR(n) if(n .ne. 0)then;call HE2(n,__LINE__);return;endif

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!   Utilities to map between global (linear algebra) to local (ghosted, physics) representations
!
!    Allocates the local work arrays and vectors
      subroutine LocalFormCreate(dm,lf,ierr)
      use mex37f90
      use mex37f90interfaces
      implicit none
      DMComposite  dm
      type(LocalForm) lf
      PetscErrorCode ierr

      call DMCompositeGetEntries(dm,lf%da,lf%np,lf%da,lf%np,ierr);CHKR(ierr) ! get the da's and sizes that define the unknowns
      call DAGetLocalInfof90(lf%da,lf%dainfo,ierr);CHKR(ierr)
      call DMCompositeGetLocalVectors(dm,lf%vIHX,lf%HotPool,lf%vCore,lf%ColdPool,ierr);
      CHKR(ierr)
      return 
      end

!     Updates the (ghosted) IHX and Core arrays from the global vector x
      subroutine LocalFormUpdate(dm,x,lf,ierr)
      use mex37f90
      use mex37f90interfaces
      implicit none
      DMComposite  dm
      type(LocalForm) lf
      PetscErrorCode ierr
      Vec x

      call DMCompositeScatter(dm,x,lf%vIHX,lf%HotPool,lf%vCore,lf%ColdPool,ierr);
      CHKR(ierr)
      call DAVecGetArrayf90(lf%da,lf%vIHX,lf%IHX,ierr);CHKR(ierr)
      call DAVecGetArrayf90(lf%da,lf%vCore,lf%Core,ierr);CHKR(ierr)
      return 
      end

!     You should call this when you no longer need access to %IHX and %Core
      subroutine LocalFormRestore(dm,lf,ierr)
      use mex37f90
      use mex37f90interfaces
      implicit none
      DMComposite  dm
      type(LocalForm) lf
      PetscErrorCode ierr

      call DAVecRestoreArrayf90(lf%da,lf%vIHX,lf%IHX,ierr);CHKR(ierr)
      call DAVecRestoreArrayf90(lf%da,lf%vCore,lf%Core,ierr);CHKR(ierr)
      return 
      end

!     Destroys the data structure 
      subroutine LocalFormDestroy(dm,lf,ierr)
      use mex37f90
      use mex37f90interfaces
      implicit none
      DMComposite  dm
      type(LocalForm) lf
      PetscErrorCode ierr

      call DMCompositeRestoreLocalVectors(dm,lf%vIHX,lf%HotPool,lf%vCore,lf%ColdPool,ierr);
      CHKR(ierr)
      return 
      end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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

      program ex37f90
      use mex37f90
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
      call DADestroy(da,ierr);CHKR(ierr)
      call DMCompositeAddArray(dm,0,app%np,ierr);CHKR(ierr)
      call DMCompositeAddDM(dm,da,ierr);CHKR(ierr)
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

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!     The physics  :-)

      subroutine FormFunction(snes,x,f,dmmg,ierr)
      use mex37f90
      use mex37f90interfaces

      SNES snes
      Vec x,f
      DMMG dmmg
      PetscErrorCode ierr

      DMComposite dm
      Vec  fvc1,fvc2
      PetscInt np,i
      DA da
      type(poolfield), pointer :: fHotPool,fColdPool
      type(channelfield), pointer :: fIHX(:),fCore(:)
      type(appctx), pointer :: app
      PetscMPIInt rank
      type(LocalForm) xlf,xoldlf

      double precision dhhpl, mult, dhcpl, phpco, pcpio, pcpci, phpii

      logical debug

      debug = .false.

      call DMMGGetUser(dmmg,app,ierr);CHKR(ierr)   ! access user context

      call DMMGGetDM(dmmg,dm,ierr);CHKR(ierr)      ! access the grid information

      call LocalFormCreate(dm,xlf,ierr)            ! Access the local parts of x
      call LocalFormUpdate(dm,x,xlf,ierr)

!       Access the global (non-ghosted) parts of f
      call DMCompositeGetEntries(dm,da,np,da,np,ierr);CHKR(ierr) ! get the da's and sizes that define the unknowns
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
      call LocalFormCreate(dm,xoldlf,ierr)            ! Access the local parts of xold
      call LocalFormUpdate(dm,app%xold,xoldlf,ierr)

      call MPI_Comm_rank(app%comm,rank,ierr);CHKR(ierr)
!      fPool only lives on zeroth process, ghosted xPool lives on all processes
      if (rank == 0) then
!
!  compute velocity into hotpool from core
!
         dhhpl = app%dhhpl0 + ( (xlf%HotPool%vol - app%hpvol0) / app%ahp )
         phpco = app%P0 + ( app%rho * app%grav * (dhhpl - app%dhco) )
         mult = app%dt / (app%dxc * app%rho)
         fHotPool%vel = xlf%HotPool%vel - (mult * (xlf%Core(app%nxc-1)%press - phpco) ) + (abs(xlf%HotPool%vel) * xlf%HotPool%vel)
!
! compute change in hot pool volume
!
         fHotPool%vol = xlf%HotPool%vol - app%hpvol0 - (app%dt * app%acore * (xlf%HotPool%vel -xlf%ColdPool%vel) )
!
!  compute velocity into coldpool from IHX
!
         dhcpl = app%dhcpl0 + ( (xlf%ColdPool%vol - app%cpvol0) / app%acp )
         pcpio = app%P0 + ( app%rho * app%grav * (dhcpl - app%dhio) )
         mult = app%dt / (app%dxc * app%rho)
         fColdPool%vel = xlf%ColdPool%vel-(mult*(xlf%IHX(app%nxc-1)%press-pcpio))+(abs(xlf%ColdPool%vel)*xlf%ColdPool%vel)
!
!  compute the change in cold pool volume
!
         fColdPool%vol = xlf%ColdPool%vol - app%cpvol0 - (app%dt * app%aihx * (xlf%ColdPool%vel - xlf%HotPool%vel) )
      endif
!
!  Compute the pressure distribution in IHX and core
!
      pcpci = app%P0 + ( app%rho * app%grav * (dhcpl - app%dhci) )
      phpii = app%P0 + ( app%rho * app%grav * (dhhpl - app%dhii) )

      do i=xlf%dainfo%xs,xlf%dainfo%xs+xlf%dainfo%xm-1

        fIHX(i)%press = xlf%IHX(i)%press - phpii - (app%rho * app%grav * dble(i) * app%dxi)
        fCore(i)%press = xlf%Core(i)%press - pcpci + (app%rho * app%grav * dble(i) * app%dxc)

      enddo

      if (debug) then
        write(*,*)
        write(*,*) 'Hot vel,vol ',xlf%HotPool%vel,xlf%HotPool%vol
        write(*,*) 'delta p = ', xlf%Core(app%nxc-1)%press - phpco,xlf%Core(app%nxc-1)%press,phpco
        write(*,*)

        do i=xlf%dainfo%xs,xlf%dainfo%xs+xlf%dainfo%xm-1
          write(*,*) 'xlf%IHX(',i,')%press = ',xlf%IHX(i)%press
        enddo

        write(*,*)
        write(*,*) 'Cold vel,vol ',xlf%ColdPool%vel,xlf%ColdPool%vol
        write(*,*) 'delta p = ',xlf%IHX(app%nxc-1)%press - pcpio,xlf%IHX(app%nxc-1)%press, pcpio
        write(*,*)


        do i=xlf%dainfo%xs,xlf%dainfo%xs+xlf%dainfo%xm-1
          write(*,*) 'xlf%Core(',i,')%press = ',xlf%Core(i)%press
        enddo

      endif

      call DAVecRestoreArrayf90(da,fvc1,fIHX,ierr);CHKR(ierr)
      call DAVecRestoreArrayf90(da,fvc2,fCore,ierr);CHKR(ierr)
      call DMCompositeRestoreAccess(dm,f,fvc1,fHotPool,fvc2,fColdPool,ierr);CHKR(ierr)

      call LocalFormRestore(dm,xoldlf,ierr)
      call LocalFormDestroy(dm,xoldlf,ierr)

      call LocalFormRestore(dm,xlf,ierr)
      call LocalFormDestroy(dm,xlf,ierr)
      return
      end

      subroutine FormGraph(dmmg,x,view0,view1,ierr)

! --------------------------------------------------------------------------------------
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
      use mex37f90
      use mex37f90interfaces

      Vec       x,xvc1,xvc2   !,corep,ihxp
      DMMG      dmmg
      PetscErrorCode ierr
      DMComposite dm
      PetscInt np            !,i
      DA da
      type(DALocalInfof90) dainfo
      type(poolfield), pointer :: xHotPool,xColdPool
      type(channelfield), pointer :: xIHX(:),xCore(:)
      type(appctx), pointer :: app

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

      call DMMGGetUser(dmmg,app,ierr);CHKR(ierr)   ! access user context
      call DMMGGetDM(dmmg,dm,ierr);CHKR(ierr)      ! Access the grid information

      call DMCompositeGetEntries(dm,da,np,da,np,ierr);CHKR(ierr) ! get the da's and sizes that define the unknowns
      call DAGetLocalInfof90(da,dainfo,ierr);CHKR(ierr)

      call DMCompositeGetLocalVectors(dm,xvc1,xHotPool,xvc2,xColdPool,ierr);CHKR(ierr)
      call DMCompositeScatter(dm,x,xvc1,xHotPool,xvc2,xColdPool,ierr);CHKR(ierr)
      call DAVecGetArrayf90(da,xvc1,xIHX,ierr);CHKR(ierr)
      call DAVecGetArrayf90(da,xvc2,xCore,ierr);CHKR(ierr)

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

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine FormInitialGuess(dmmg,v,ierr)
      use mex37f90
      use mex37f90interfaces

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
