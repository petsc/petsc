#include "finclude/petscdef.h"

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
!
!     For Newton's method with block Jacobi (one block for 
!     each channel and one block for each pool) run with
!
!       -dmmg_iscoloring_type global -snes_mf_operator -pc_type lu 
!       -help lists all options

      program ex34f90
      use mex34f90


      DMMG             dmmg
      PetscErrorCode   ierr
      DA               da
      DMComposite      dm
      type(appctx)     app
      external         FormInitialGuess,FormFunction


      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

!      Create the composite DM object that manages the grid

      call DMCompositeCreate(PETSC_COMM_WORLD,dm,ierr)
      call DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,app%nxc,app%nc,1, &
     &                PETSC_NULL_INTEGER,da,ierr)
      call DMCompositeAddDM(dm,da,ierr)
      call DADestroy(da,ierr)
      call DMCompositeAddArray(dm,0,app%np,ierr)
      call DMCompositeAddDM(dm,da,ierr)
      call DMCompositeAddArray(dm,0,app%np,ierr)

!       Create solver object and associate it with the unknowns (on the grid)

      call DMMGCreate(PETSC_COMM_WORLD,1,PETSC_NULL_OBJECT,dmmg,ierr)
      call DMMGSetDM(dmmg,dm,ierr)
      call DMCompositeDestroy(dm,ierr)
      call DMMGSetUser(dmmg,0,app,ierr)  ! currently only one level solver
      call DMMGSetInitialGuess(dmmg,FormInitialGuess,ierr)
      call DMMGSetSNES(dmmg,FormFunction,PETSC_NULL_OBJECT,ierr)
      call DMMGSetFromOptions(dmmg,ierr)

!      Solve the nonlinear system
!
      call DMMGSolve(dmmg,ierr)

      call DMMGDestroy(dmmg,ierr)
      call PetscFinalize(ierr)
      end


      subroutine FormInitialGuess(dmmg,v,ierr)
      use mex34f90
      use mex34finterfaces

      DMMG dmmg
      Vec v
      PetscErrorCode ierr

      DMComposite dm
      Vec  vc1,vc2
      PetscInt np,i
      DA da
      type(DALocalInfof90) dainfo
      type(poolfield), pointer :: Pool1,Pool2
      type(channelfield), pointer :: v1(:),v2(:)
      type(appctx), pointer :: app
      PetscMPIInt rank

      call DMMGGetUser(dmmg,app,ierr)   ! access user context

!       Access the grid information

      call DMMGGetDM(dmmg,dm,ierr) 
      call DMCompositeGetEntries(dm,da,np,da,np,ierr) ! get the da's and sizes that define the unknowns
      call DAGetLocalInfof90(da,dainfo,ierr)

!      Acess the vector unknowns in global (nonghosted) form

      call DMCompositeGetAccess(dm,v,vc1,Pool1,vc2,Pool2,ierr)  ! 
      call DAVecGetArrayf90(da,vc1,v1,ierr)
      call DAVecGetArrayf90(da,vc2,v2,ierr)

      call MPI_Comm_rank(app%comm,rank,ierr)
!      the pools are only unknowns on process 0
      if (rank == 0) then
         Pool1%p0 = -2.0
         Pool1%p1 = -2000.0
         Pool2%p0 = -20
         Pool2%p1 = -200
      endif

!       put some random numbers as an initial guess 
      do i=dainfo%xs,dainfo%xs+dainfo%xm-1
        v1(i)%c0 = 3*i - .5
        v1(i)%c1 = 3*i - (1.d0/3.d0)
        v1(i)%c2 = 3*i - .1
      enddo 

      call DAVecRestoreArrayf90(da,vc1,v1,ierr)
      call DAVecRestoreArrayf90(da,vc2,v2,ierr)
      call DMCompositeRestoreAccess(dm,v,vc1,Pool1,vc2,Pool2,ierr)

      CHKMEMQ
      return
      end      

      subroutine FormFunction(snes,x,f,dmmg,ierr)
      use mex34f90
      use mex34finterfaces

      SNES snes
      Vec x,f
      DMMG dmmg
      PetscErrorCode ierr

      DMComposite dm
      Vec  fvc1,fvc2,xvc1,xvc2
      PetscInt np,i
      DA da
      type(DALocalInfof90) dainfo
      type(poolfield), pointer :: fPool1,fPool2
      type(poolfield), pointer :: xPool1,xPool2
      type(channelfield), pointer :: fv1(:),fv2(:),xv1(:),xv2(:)
      type(appctx), pointer :: app
      PetscMPIInt rank

      call DMMGGetUser(dmmg,app,ierr)   ! access user context

!         Access the grid information

      call DMMGGetDM(dmmg,dm,ierr)  
      call DMCompositeGetEntries(dm,da,np,da,np,ierr) ! get the da's and sizes that define the unknowns
      call DAGetLocalInfof90(da,dainfo,ierr)

!        Access the local (ghosted) parts of x

      call DMCompositeGetLocalVectors(dm,xvc1,xPool1,xvc2,xPool2,ierr)  
      call DMCompositeScatter(dm,x,xvc1,xPool1,xvc2,xPool2,ierr)
      call DAVecGetArrayf90(da,xvc1,xv1,ierr)
      call DAVecGetArrayf90(da,xvc2,xv2,ierr)

!       Access the global (non-ghosted) parts of f

      call DMCompositeGetAccess(dm,f,fvc1,fPool1,fvc2,fPool2,ierr)
      call DAVecGetArrayf90(da,fvc1,fv1,ierr)
      call DAVecGetArrayf90(da,fvc2,fv2,ierr)


      call MPI_Comm_rank(app%comm,rank,ierr)
!      fPool only lives on zeroth process, ghosted xPool lives on all processes
      if (rank == 0) then
         fPool1%p0 = xPool1%p0
         fPool1%p1 = xPool1%p1
         fPool2%p0 = xPool2%p0
         fPool2%p1 = xPool2%p1
      endif

!        Replace with some difference scheme
      do i=dainfo%xs,dainfo%xs+dainfo%xm-1
        fv1(i)%c0 = xv1(i)%c0
        fv1(i)%c1 = xv1(i)%c1
        fv1(i)%c2 = xv1(i)%c2

        fv2(i)%c0 = xv2(i)%c0
        fv2(i)%c1 = xv2(i)%c1
        fv2(i)%c2 = xv2(i)%c2
      enddo 

      call DAVecRestoreArrayf90(da,xvc1,xv1,ierr)
      call DAVecRestoreArrayf90(da,xvc2,xv2,ierr)
      call DMCompositeRestoreLocalVectors(dm,xvc1,xPool1,xvc2,xPool2,   &
     &     ierr)  

      call DAVecRestoreArrayf90(da,fvc1,fv1,ierr)
      call DAVecRestoreArrayf90(da,fvc2,fv2,ierr)
      call DMCompositeRestoreAccess(dm,f,fvc1,fPool1,fvc2,fPool2,ierr)

      return
      end      

