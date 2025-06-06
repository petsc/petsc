!
!
!  This example demonstrates basic use of the SNES Fortran interface.
!
!
        module ex12fmodule
#include <petsc/finclude/petscsnes.h>
        use petscsnes
        type User
          DM  da
          Vec F
          Vec xl
          MPI_Comm comm
          PetscInt N
        end type User
        save
        type monctx
        PetscInt :: its,lag
        end type monctx
      end module

! ---------------------------------------------------------------------
!  Subroutine FormMonitor
!  This function lets up keep track of the SNES progress at each step
!  In this routine, we determine when the Jacobian is rebuilt with the parameter 'jag'
!
!  Input Parameters:
!    snes    - SNES nonlinear solver context
!    its     - current nonlinear iteration, starting from a call of SNESSolve()
!    norm    - 2-norm of current residual (may be approximate)
!    snesm - monctx designed module (included in Snesmmod)
! ---------------------------------------------------------------------
      subroutine FormMonitor(snes,its,norm,snesm,ierr)
      use ex12fmodule
      implicit none

      SNES ::           snes
      PetscInt ::       its,one,mone
      PetscScalar ::    norm
      type(monctx) ::   snesm
      PetscErrorCode :: ierr

!      write(6,*) ' '
!      write(6,*) '    its ',its,snesm%its,'lag',
!     &            snesm%lag
!      call flush(6)
      if (mod(snesm%its,snesm%lag).eq.0) then
        one = 1
        PetscCall(SNESSetLagJacobian(snes,one,ierr))  ! build jacobian
      else
        mone = -1
        PetscCall(SNESSetLagJacobian(snes,mone,ierr)) ! do NOT build jacobian
      endif
      snesm%its = snesm%its + 1
      end subroutine FormMonitor

!  Note: Any user-defined Fortran routines (such as FormJacobian)
!  MUST be declared as external.
!

      program main
      use ex12fmodule
      implicit none
      type(User) ctx
      PetscMPIInt rank,size
      PetscErrorCode ierr
      PetscInt N,start,end,nn,i
      PetscInt ii,its,i1,i0,i3
      PetscBool  flg
      SNES             snes
      Mat              J
      Vec              x,r,u
      PetscScalar      xp,FF,UU,h
      character*(10)   matrixname
      external         FormJacobian,FormFunction
      external         formmonitor
      type(monctx) :: snesm

      PetscCallA(PetscInitialize(ierr))
      i1 = 1
      i0 = 0
      i3 = 3
      N  = 10
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',N,flg,ierr))
      h = 1.0/real(N-1)
      ctx%N = N
      ctx%comm = PETSC_COMM_WORLD

      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))
      PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD,size,ierr))

! Set up data structures
      PetscCallA(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,N,i1,i1,PETSC_NULL_INTEGER_ARRAY,ctx%da,ierr))
      PetscCallA(DMSetFromOptions(ctx%da,ierr))
      PetscCallA(DMSetUp(ctx%da,ierr))
      PetscCallA(DMCreateGlobalVector(ctx%da,x,ierr))
      PetscCallA(DMCreateLocalVector(ctx%da,ctx%xl,ierr))

      PetscCallA(PetscObjectSetName(x,'Approximate Solution',ierr))
      PetscCallA(VecDuplicate(x,r,ierr))
      PetscCallA(VecDuplicate(x,ctx%F,ierr))
      PetscCallA(VecDuplicate(x,U,ierr))
      PetscCallA(PetscObjectSetName(U,'Exact Solution',ierr))

      PetscCallA(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,N,i3,PETSC_NULL_INTEGER_ARRAY,i0,PETSC_NULL_INTEGER_ARRAY,J,ierr))
      PetscCallA(MatSetOption(J,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE,ierr))
      PetscCallA(MatGetType(J,matrixname,ierr))

! Store right-hand side of PDE and exact solution
      PetscCallA(VecGetOwnershipRange(x,start,end,ierr))
      xp = h*start
      nn = end - start
      ii = start
      do 10, i=0,nn-1
        FF = 6.0*xp + (xp+1.e-12)**6.e0
        UU = xp*xp*xp
        PetscCallA(VecSetValues(ctx%F,i1,[ii],[FF],INSERT_VALUES,ierr))
        PetscCallA(VecSetValues(U,i1,[ii],[UU],INSERT_VALUES,ierr))
        xp = xp + h
        ii = ii + 1
 10   continue
      PetscCallA(VecAssemblyBegin(ctx%F,ierr))
      PetscCallA(VecAssemblyEnd(ctx%F,ierr))
      PetscCallA(VecAssemblyBegin(U,ierr))
      PetscCallA(VecAssemblyEnd(U,ierr))

! Create nonlinear solver
      PetscCallA(SNESCreate(PETSC_COMM_WORLD,snes,ierr))

! Set various routines and options
      PetscCallA(SNESSetFunction(snes,r,FormFunction,ctx,ierr))
      PetscCallA(SNESSetJacobian(snes,J,J,FormJacobian,ctx,ierr))

      snesm%its = 0
      PetscCallA(SNESGetLagJacobian(snes,snesm%lag,ierr))
      PetscCallA(SNESMonitorSet(snes,FormMonitor,snesm,PETSC_NULL_FUNCTION,ierr))
      PetscCallA(SNESSetFromOptions(snes,ierr))

! Solve nonlinear system
      PetscCallA(FormInitialGuess(snes,x,ierr))
      PetscCallA(SNESSolve(snes,PETSC_NULL_VEC,x,ierr))
      PetscCallA(SNESGetIterationNumber(snes,its,ierr))

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.
      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(VecDestroy(ctx%xl,ierr))
      PetscCallA(VecDestroy(r,ierr))
      PetscCallA(VecDestroy(U,ierr))
      PetscCallA(VecDestroy(ctx%F,ierr))
      PetscCallA(MatDestroy(J,ierr))
      PetscCallA(SNESDestroy(snes,ierr))
      PetscCallA(DMDestroy(ctx%da,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

! --------------------  Evaluate Function F(x) ---------------------

      subroutine FormFunction(snes,x,f,ctx,ierr)
      use ex12fmodule
      implicit none
      SNES             snes
      Vec              x,f
      type(User) ctx
      PetscMPIInt  rank,size,zero
      PetscInt i,s,n
      PetscErrorCode ierr
      PetscScalar      h,d
      PetscScalar,pointer :: vf2(:),vxx(:),vff(:)

      zero = 0
      PetscCallMPI(MPI_Comm_rank(ctx%comm,rank,ierr))
      PetscCallMPI(MPI_Comm_size(ctx%comm,size,ierr))
      h     = 1.0/(real(ctx%N) - 1.0)
      PetscCall(DMGlobalToLocalBegin(ctx%da,x,INSERT_VALUES,ctx%xl,ierr))
      PetscCall(DMGlobalToLocalEnd(ctx%da,x,INSERT_VALUES,ctx%xl,ierr))

      PetscCall(VecGetLocalSize(ctx%xl,n,ierr))
      if (n .gt. 1000) then
        print*, 'Local work array not big enough'
        call MPI_Abort(PETSC_COMM_WORLD,zero,ierr)
      endif

      PetscCall(VecGetArrayRead(ctx%xl,vxx,ierr))
      PetscCall(VecGetArray(f,vff,ierr))
      PetscCall(VecGetArray(ctx%F,vF2,ierr))

      d = h*h

!
!  Note that the array vxx() was obtained from a ghosted local vector
!  ctx%xl while the array vff() was obtained from the non-ghosted parallel
!  vector F. This is why there is a need for shift variable s. Since vff()
!  does not have locations for the ghost variables we need to index in it
!  slightly different then indexing into vxx(). For example on processor
!  1 (the second processor)
!
!        xx(1)        xx(2)             xx(3)             .....
!      ^^^^^^^        ^^^^^             ^^^^^
!      ghost value   1st local value   2nd local value
!
!                      ff(1)             ff(2)
!                     ^^^^^^^           ^^^^^^^
!                    1st local value   2nd local value
!
       if (rank .eq. 0) then
        s = 0
        vff(1) = vxx(1)
      else
        s = 1
      endif

      do 10 i=1,n-2
       vff(i-s+1) = d*(vxx(i) - 2.0*vxx(i+1) + vxx(i+2)) + vxx(i+1)*vxx(i+1) - vF2(i-s+1)
 10   continue

      if (rank .eq. size-1) then
        vff(n-s) = vxx(n) - 1.0
      endif

      PetscCall(VecRestoreArray(f,vff,ierr))
      PetscCall(VecRestoreArrayRead(ctx%xl,vxx,ierr))
      PetscCall(VecRestoreArray(ctx%F,vF2,ierr))
      end

! --------------------  Form initial approximation -----------------

      subroutine FormInitialGuess(snes,x,ierr)
      use ex12fmodule
      implicit none

      PetscErrorCode   ierr
      Vec              x
      SNES             snes
      PetscScalar      five

      five = .5
      PetscCall(VecSet(x,five,ierr))
      end

! --------------------  Evaluate Jacobian --------------------

      subroutine FormJacobian(snes,x,jac,B,ctx,ierr)
      use ex12fmodule
      implicit none

      SNES             snes
      Vec              x
      Mat              jac,B
      type(User) ctx
      PetscInt  ii,istart,iend
      PetscInt  i,j,n,end,start,i1
      PetscErrorCode ierr
      PetscMPIInt rank,size
      PetscScalar      d,A,h
      PetscScalar,pointer :: vxx(:)

      i1 = 1
      h = 1.0/(real(ctx%N) - 1.0)
      d = h*h
      PetscCallMPI(MPI_Comm_rank(ctx%comm,rank,ierr))
      PetscCallMPI(MPI_Comm_size(ctx%comm,size,ierr))

      PetscCall(VecGetArrayRead(x,vxx,ierr))
      PetscCall(VecGetOwnershipRange(x,start,end,ierr))
      n = end - start

      if (rank .eq. 0) then
        A = 1.0
        PetscCall(MatSetValues(jac,i1,[start],i1,[start],[A],INSERT_VALUES,ierr))
        istart = 1
      else
        istart = 0
      endif
      if (rank .eq. size-1) then
        i = INT(ctx%N-1)
        A = 1.0
        PetscCall(MatSetValues(jac,i1,[i],i1,[i],[A],INSERT_VALUES,ierr))
        iend = n-1
      else
        iend = n
      endif
      do 10 i=istart,iend-1
        ii = i + start
        j = start + i - 1
        PetscCall(MatSetValues(jac,i1,[ii],i1,[j],[d],INSERT_VALUES,ierr))
        j = start + i + 1
        PetscCall(MatSetValues(jac,i1,[ii],i1,[j],[d],INSERT_VALUES,ierr))
        A = -2.0*d + 2.0*vxx(i+1)
        PetscCall(MatSetValues(jac,i1,[ii],i1,[ii],[A],INSERT_VALUES,ierr))
 10   continue
      PetscCall(VecRestoreArrayRead(x,vxx,ierr))
      PetscCall(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY,ierr))
      PetscCall(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY,ierr))
      end

!/*TEST
!
!   test:
!      nsize: 2
!      args: -ksp_gmres_cgs_refinement_type refine_always -n 10 -snes_monitor_short
!      output_file: output/ex12_1.out
!
!TEST*/
