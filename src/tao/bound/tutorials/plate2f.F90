!  Program usage: mpiexec -n <proc> plate2f [all TAO options]
!
!  This example demonstrates use of the TAO package to solve a bound constrained
!  minimization problem.  This example is based on a problem from the
!  MINPACK-2 test suite.  Given a rectangular 2-D domain and boundary values
!  along the edges of the domain, the objective is to find the surface
!  with the minimal area that satisfies the boundary conditions.
!  The command line options are:
!    -mx <xg>, where <xg> = number of grid points in the 1st coordinate direction
!    -my <yg>, where <yg> = number of grid points in the 2nd coordinate direction
!    -bmx <bxg>, where <bxg> = number of grid points under plate in 1st direction
!    -bmy <byg>, where <byg> = number of grid points under plate in 2nd direction
!    -bheight <ht>, where <ht> = height of the plate
!

      module mymodule
#include "petsc/finclude/petscdmda.h"
#include "petsc/finclude/petsctao.h"
      use petscdmda
      use petsctao

      Vec              localX, localV
      Vec              Top, Left
      Vec              Right, Bottom
      DM               dm
      PetscReal      bheight
      PetscInt         bmx, bmy
      PetscInt         mx, my, Nx, Ny, N
      end module

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Variables:
!    (common from plate2f.h):
!    Nx, Ny           number of processors in x- and y- directions
!    mx, my           number of grid points in x,y directions
!    N    global dimension of vector
      use mymodule
      implicit none

      PetscErrorCode   ierr          ! used to check for functions returning nonzeros
      Vec              x             ! solution vector
      PetscInt         m             ! number of local elements in vector
      Tao              tao           ! Tao solver context
      Mat              H             ! Hessian matrix
      ISLocalToGlobalMapping isltog  ! local to global mapping object
      PetscBool        flg
      PetscInt         i1,i3,i7

      external FormFunctionGradient
      external FormHessian
      external MSA_BoundaryConditions
      external MSA_Plate
      external MSA_InitialPoint
! Initialize Tao

      i1=1
      i3=3
      i7=7

      PetscCallA(PetscInitialize(ierr))

! Specify default dimensions of the problem
      mx = 10
      my = 10
      bheight = 0.1

! Check for any command line arguments that override defaults

      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-mx',mx,flg,ierr))
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-my',my,flg,ierr))

      bmx = mx/2
      bmy = my/2

      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-bmx',bmx,flg,ierr))
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-bmy',bmy,flg,ierr))
      PetscCallA(PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-bheight',bheight,flg,ierr))

! Calculate any derived values from parameters
      N = mx*my

! Let Petsc determine the dimensions of the local vectors
      Nx = PETSC_DECIDE
      NY = PETSC_DECIDE

! A two dimensional distributed array will help define this problem, which
! derives from an elliptic PDE on a two-dimensional domain.  From the
! distributed array, create the vectors

      PetscCallA(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,mx,my,Nx,Ny,i1,i1,PETSC_NULL_INTEGER,PETSC_NULL_INTEGER,dm,ierr))
      PetscCallA(DMSetFromOptions(dm,ierr))
      PetscCallA(DMSetUp(dm,ierr))

! Extract global and local vectors from DM; The local vectors are
! used solely as work space for the evaluation of the function,
! gradient, and Hessian.  Duplicate for remaining vectors that are
! the same types.

      PetscCallA(DMCreateGlobalVector(dm,x,ierr))
      PetscCallA(DMCreateLocalVector(dm,localX,ierr))
      PetscCallA(VecDuplicate(localX,localV,ierr))

! Create a matrix data structure to store the Hessian.
! Here we (optionally) also associate the local numbering scheme
! with the matrix so that later we can use local indices for matrix
! assembly

      PetscCallA(VecGetLocalSize(x,m,ierr))
      PetscCallA(MatCreateAIJ(PETSC_COMM_WORLD,m,m,N,N,i7,PETSC_NULL_INTEGER,i3,PETSC_NULL_INTEGER,H,ierr))

      PetscCallA(MatSetOption(H,MAT_SYMMETRIC,PETSC_TRUE,ierr))
      PetscCallA(DMGetLocalToGlobalMapping(dm,isltog,ierr))
      PetscCallA(MatSetLocalToGlobalMapping(H,isltog,isltog,ierr))

! The Tao code begins here
! Create TAO solver and set desired solution method.
! This problems uses bounded variables, so the
! method must either be 'tao_tron' or 'tao_blmvm'

      PetscCallA(TaoCreate(PETSC_COMM_WORLD,tao,ierr))
      PetscCallA(TaoSetType(tao,TAOBLMVM,ierr))

!     Set minimization function and gradient, hessian evaluation functions

      PetscCallA(TaoSetObjectiveAndGradient(tao,PETSC_NULL_VEC,FormFunctionGradient,0,ierr))

      PetscCallA(TaoSetHessian(tao,H,H,FormHessian,0, ierr))

! Set Variable bounds
      PetscCallA(MSA_BoundaryConditions(ierr))
      PetscCallA(TaoSetVariableBoundsRoutine(tao,MSA_Plate,0,ierr))

! Set the initial solution guess
      PetscCallA(MSA_InitialPoint(x, ierr))
      PetscCallA(TaoSetSolution(tao,x,ierr))

! Check for any tao command line options
      PetscCallA(TaoSetFromOptions(tao,ierr))

! Solve the application
      PetscCallA(TaoSolve(tao,ierr))

! Free TAO data structures
      PetscCallA(TaoDestroy(tao,ierr))

! Free PETSc data structures
      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(VecDestroy(Top,ierr))
      PetscCallA(VecDestroy(Bottom,ierr))
      PetscCallA(VecDestroy(Left,ierr))
      PetscCallA(VecDestroy(Right,ierr))
      PetscCallA(MatDestroy(H,ierr))
      PetscCallA(VecDestroy(localX,ierr))
      PetscCallA(VecDestroy(localV,ierr))
      PetscCallA(DMDestroy(dm,ierr))

! Finalize TAO

      PetscCallA(PetscFinalize(ierr))

      end

! ---------------------------------------------------------------------
!
!  FormFunctionGradient - Evaluates function f(X).
!
!  Input Parameters:
!  tao   - the Tao context
!  X     - the input vector
!  dummy - optional user-defined context, as set by TaoSetFunction()
!          (not used here)
!
!  Output Parameters:
!  fcn     - the newly evaluated function
!  G       - the gradient vector
!  info  - error code
!

      subroutine FormFunctionGradient(tao,X,fcn,G,dummy,ierr)
      use mymodule
      implicit none

! Input/output variables

      Tao        tao
      PetscReal      fcn
      Vec              X, G
      PetscErrorCode   ierr
      PetscInt         dummy

      PetscInt         i,j,row
      PetscInt         xs, xm
      PetscInt         gxs, gxm
      PetscInt         ys, ym
      PetscInt         gys, gym
      PetscReal      ft,zero,hx,hy,hydhx,hxdhy
      PetscReal      area,rhx,rhy
      PetscReal      f1,f2,f3,f4,f5,f6,d1,d2,d3
      PetscReal      d4,d5,d6,d7,d8
      PetscReal      df1dxc,df2dxc,df3dxc,df4dxc
      PetscReal      df5dxc,df6dxc
      PetscReal      xc,xl,xr,xt,xb,xlt,xrb

! PETSc's VecGetArray acts differently in Fortran than it does in C.
! Calling VecGetArray((Vec) X, (PetscReal) x_array(0:1), (PetscOffset) x_index, ierr)
! will return an array of doubles referenced by x_array offset by x_index.
!  i.e.,  to reference the kth element of X, use x_array(k + x_index).
! Notice that by declaring the arrays with range (0:1), we are using the C 0-indexing practice.
      PetscReal      g_v(0:1),x_v(0:1)
      PetscReal      top_v(0:1),left_v(0:1)
      PetscReal      right_v(0:1),bottom_v(0:1)
      PetscOffset      g_i,left_i,right_i
      PetscOffset      bottom_i,top_i,x_i

      ft = 0.0
      zero = 0.0
      hx = 1.0/real(mx + 1)
      hy = 1.0/real(my + 1)
      hydhx = hy/hx
      hxdhy = hx/hy
      area = 0.5 * hx * hy
      rhx = real(mx) + 1.0
      rhy = real(my) + 1.0

! Get local mesh boundaries
      PetscCall(DMDAGetCorners(dm,xs,ys,PETSC_NULL_INTEGER,xm,ym,PETSC_NULL_INTEGER,ierr))
      PetscCall(DMDAGetGhostCorners(dm,gxs,gys,PETSC_NULL_INTEGER,gxm,gym,PETSC_NULL_INTEGER,ierr))

! Scatter ghost points to local vector
      PetscCall(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,localX,ierr))
      PetscCall(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,localX,ierr))

! Initialize the vector to zero
      PetscCall(VecSet(localV,zero,ierr))

! Get arrays to vector data (See note above about using VecGetArray in Fortran)
      PetscCall(VecGetArray(localX,x_v,x_i,ierr))
      PetscCall(VecGetArray(localV,g_v,g_i,ierr))
      PetscCall(VecGetArray(Top,top_v,top_i,ierr))
      PetscCall(VecGetArray(Bottom,bottom_v,bottom_i,ierr))
      PetscCall(VecGetArray(Left,left_v,left_i,ierr))
      PetscCall(VecGetArray(Right,right_v,right_i,ierr))

! Compute function over the locally owned part of the mesh
      do j = ys,ys+ym-1
         do i = xs,xs+xm-1
            row = (j-gys)*gxm + (i-gxs)
            xc = x_v(row+x_i)
            xt = xc
            xb = xc
            xr = xc
            xl = xc
            xrb = xc
            xlt = xc

            if (i .eq. 0) then !left side
               xl = left_v(j - ys + 1 + left_i)
               xlt = left_v(j - ys + 2 + left_i)
            else
               xl = x_v(row - 1 + x_i)
            endif

            if (j .eq. 0) then !bottom side
               xb = bottom_v(i - xs + 1 + bottom_i)
               xrb = bottom_v(i - xs + 2 + bottom_i)
            else
               xb = x_v(row - gxm + x_i)
            endif

            if (i + 1 .eq. gxs + gxm) then !right side
               xr = right_v(j - ys + 1 + right_i)
               xrb = right_v(j - ys + right_i)
            else
               xr = x_v(row + 1 + x_i)
            endif

            if (j + 1 .eq. gys + gym) then !top side
               xt = top_v(i - xs + 1 + top_i)
               xlt = top_v(i - xs + top_i)
            else
               xt = x_v(row + gxm + x_i)
            endif

            if ((i .gt. gxs) .and. (j + 1 .lt. gys + gym)) then
               xlt = x_v(row - 1 + gxm + x_i)
            endif

            if ((j .gt. gys) .and. (i + 1 .lt. gxs + gxm)) then
               xrb = x_v(row + 1 - gxm + x_i)
            endif

            d1 = xc-xl
            d2 = xc-xr
            d3 = xc-xt
            d4 = xc-xb
            d5 = xr-xrb
            d6 = xrb-xb
            d7 = xlt-xl
            d8 = xt-xlt

            df1dxc = d1 * hydhx
            df2dxc = d1 * hydhx + d4 * hxdhy
            df3dxc = d3 * hxdhy
            df4dxc = d2 * hydhx + d3 * hxdhy
            df5dxc = d2 * hydhx
            df6dxc = d4 * hxdhy

            d1 = d1 * rhx
            d2 = d2 * rhx
            d3 = d3 * rhy
            d4 = d4 * rhy
            d5 = d5 * rhy
            d6 = d6 * rhx
            d7 = d7 * rhy
            d8 = d8 * rhx

            f1 = sqrt(1.0 + d1*d1 + d7*d7)
            f2 = sqrt(1.0 + d1*d1 + d4*d4)
            f3 = sqrt(1.0 + d3*d3 + d8*d8)
            f4 = sqrt(1.0 + d3*d3 + d2*d2)
            f5 = sqrt(1.0 + d2*d2 + d5*d5)
            f6 = sqrt(1.0 + d4*d4 + d6*d6)

            ft = ft + f2 + f4

            df1dxc = df1dxc / f1
            df2dxc = df2dxc / f2
            df3dxc = df3dxc / f3
            df4dxc = df4dxc / f4
            df5dxc = df5dxc / f5
            df6dxc = df6dxc / f6

            g_v(row + g_i) = 0.5 * (df1dxc + df2dxc + df3dxc + df4dxc + df5dxc + df6dxc)
         enddo
      enddo

! Compute triangular areas along the border of the domain.
      if (xs .eq. 0) then  ! left side
         do j=ys,ys+ym-1
            d3 = (left_v(j-ys+1+left_i) - left_v(j-ys+2+left_i)) * rhy
            d2 = (left_v(j-ys+1+left_i) - x_v((j-gys)*gxm + x_i)) * rhx
            ft = ft + sqrt(1.0 + d3*d3 + d2*d2)
         enddo
      endif

      if (ys .eq. 0) then !bottom side
         do i=xs,xs+xm-1
            d2 = (bottom_v(i+1-xs+bottom_i)-bottom_v(i-xs+2+bottom_i)) * rhx
            d3 = (bottom_v(i-xs+1+bottom_i)-x_v(i-gxs+x_i))*rhy
            ft = ft + sqrt(1.0 + d3*d3 + d2*d2)
         enddo
      endif

      if (xs + xm .eq. mx) then ! right side
         do j=ys,ys+ym-1
            d1 = (x_v((j+1-gys)*gxm-1+x_i)-right_v(j-ys+1+right_i))*rhx
            d4 = (right_v(j-ys+right_i) - right_v(j-ys+1+right_i))*rhy
            ft = ft + sqrt(1.0 + d1*d1 + d4*d4)
         enddo
      endif

      if (ys + ym .eq. my) then
         do i=xs,xs+xm-1
            d1 = (x_v((gym-1)*gxm+i-gxs+x_i) - top_v(i-xs+1+top_i))*rhy
            d4 = (top_v(i-xs+1+top_i) - top_v(i-xs+top_i))*rhx
            ft = ft + sqrt(1.0 + d1*d1 + d4*d4)
         enddo
      endif

      if ((ys .eq. 0) .and. (xs .eq. 0)) then
         d1 = (left_v(0 + left_i) - left_v(1 + left_i)) * rhy
         d2 = (bottom_v(0+bottom_i)-bottom_v(1+bottom_i))*rhx
         ft = ft + sqrt(1.0 + d1*d1 + d2*d2)
      endif

      if ((ys + ym .eq. my) .and. (xs + xm .eq. mx)) then
         d1 = (right_v(ym+1+right_i) - right_v(ym+right_i))*rhy
         d2 = (top_v(xm+1+top_i) - top_v(xm + top_i))*rhx
         ft = ft + sqrt(1.0 + d1*d1 + d2*d2)
      endif

      ft = ft * area
      PetscCallMPI(MPI_Allreduce(ft,fcn,1,MPIU_SCALAR,MPIU_SUM,PETSC_COMM_WORLD,ierr))

! Restore vectors
      PetscCall(VecRestoreArray(localX,x_v,x_i,ierr))
      PetscCall(VecRestoreArray(localV,g_v,g_i,ierr))
      PetscCall(VecRestoreArray(Left,left_v,left_i,ierr))
      PetscCall(VecRestoreArray(Top,top_v,top_i,ierr))
      PetscCall(VecRestoreArray(Bottom,bottom_v,bottom_i,ierr))
      PetscCall(VecRestoreArray(Right,right_v,right_i,ierr))

! Scatter values to global vector
      PetscCall(DMLocalToGlobalBegin(dm,localV,INSERT_VALUES,G,ierr))
      PetscCall(DMLocalToGlobalEnd(dm,localV,INSERT_VALUES,G,ierr))

      PetscCall(PetscLogFlops(70.0d0*xm*ym,ierr))

      return
      end  !FormFunctionGradient

! ----------------------------------------------------------------------------
!
!
!   FormHessian - Evaluates Hessian matrix.
!
!   Input Parameters:
!.  tao  - the Tao context
!.  X    - input vector
!.  dummy  - not used
!
!   Output Parameters:
!.  Hessian    - Hessian matrix
!.  Hpc    - optionally different preconditioning matrix
!.  flag - flag indicating matrix structure
!
!   Notes:
!   Due to mesh point reordering with DMs, we must always work
!   with the local mesh points, and then transform them to the new
!   global numbering with the local-to-global mapping.  We cannot work
!   directly with the global numbers for the original uniprocessor mesh!
!
!      MatSetValuesLocal(), using the local ordering (including
!         ghost points!)
!         - Then set matrix entries using the local ordering
!           by calling MatSetValuesLocal()

      subroutine FormHessian(tao, X, Hessian, Hpc, dummy, ierr)
      use mymodule
      implicit none

      Tao     tao
      Vec            X
      Mat            Hessian,Hpc
      PetscInt       dummy
      PetscErrorCode ierr

      PetscInt       i,j,k,row
      PetscInt       xs,xm,gxs,gxm
      PetscInt       ys,ym,gys,gym
      PetscInt       col(0:6)
      PetscReal    hx,hy,hydhx,hxdhy,rhx,rhy
      PetscReal    f1,f2,f3,f4,f5,f6,d1,d2,d3
      PetscReal    d4,d5,d6,d7,d8
      PetscReal    xc,xl,xr,xt,xb,xlt,xrb
      PetscReal    hl,hr,ht,hb,hc,htl,hbr

! PETSc's VecGetArray acts differently in Fortran than it does in C.
! Calling VecGetArray((Vec) X, (PetscReal) x_array(0:1), (PetscOffset) x_index, ierr)
! will return an array of doubles referenced by x_array offset by x_index.
!  i.e.,  to reference the kth element of X, use x_array(k + x_index).
! Notice that by declaring the arrays with range (0:1), we are using the C 0-indexing practice.
      PetscReal   right_v(0:1),left_v(0:1)
      PetscReal   bottom_v(0:1),top_v(0:1)
      PetscReal   x_v(0:1)
      PetscOffset   x_i,right_i,left_i
      PetscOffset   bottom_i,top_i
      PetscReal   v(0:6)
      PetscBool     assembled
      PetscInt      i1

      i1=1

! Set various matrix options
      PetscCall(MatSetOption(Hessian,MAT_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE,ierr))

! Get local mesh boundaries
      PetscCall(DMDAGetCorners(dm,xs,ys,PETSC_NULL_INTEGER,xm,ym,PETSC_NULL_INTEGER,ierr))
      PetscCall(DMDAGetGhostCorners(dm,gxs,gys,PETSC_NULL_INTEGER,gxm,gym,PETSC_NULL_INTEGER,ierr))

! Scatter ghost points to local vectors
      PetscCall(DMGlobalToLocalBegin(dm,X,INSERT_VALUES,localX,ierr))
      PetscCall(DMGlobalToLocalEnd(dm,X,INSERT_VALUES,localX,ierr))

! Get pointers to vector data (see note on Fortran arrays above)
      PetscCall(VecGetArray(localX,x_v,x_i,ierr))
      PetscCall(VecGetArray(Top,top_v,top_i,ierr))
      PetscCall(VecGetArray(Bottom,bottom_v,bottom_i,ierr))
      PetscCall(VecGetArray(Left,left_v,left_i,ierr))
      PetscCall(VecGetArray(Right,right_v,right_i,ierr))

! Initialize matrix entries to zero
      PetscCall(MatAssembled(Hessian,assembled,ierr))
      if (assembled .eqv. PETSC_TRUE) PetscCall(MatZeroEntries(Hessian,ierr))

      rhx = real(mx) + 1.0
      rhy = real(my) + 1.0
      hx = 1.0/rhx
      hy = 1.0/rhy
      hydhx = hy/hx
      hxdhy = hx/hy
! compute Hessian over the locally owned part of the mesh

      do  i=xs,xs+xm-1
         do  j=ys,ys+ym-1
            row = (j-gys)*gxm + (i-gxs)

            xc = x_v(row + x_i)
            xt = xc
            xb = xc
            xr = xc
            xl = xc
            xrb = xc
            xlt = xc

            if (i .eq. gxs) then   ! Left side
               xl = left_v(left_i + j - ys + 1)
               xlt = left_v(left_i + j - ys + 2)
            else
               xl = x_v(x_i + row -1)
            endif

            if (j .eq. gys) then ! bottom side
               xb = bottom_v(bottom_i + i - xs + 1)
               xrb = bottom_v(bottom_i + i - xs + 2)
            else
               xb = x_v(x_i + row - gxm)
            endif

            if (i+1 .eq. gxs + gxm) then !right side
               xr = right_v(right_i + j - ys + 1)
               xrb = right_v(right_i + j - ys)
            else
               xr = x_v(x_i + row + 1)
            endif

            if (j+1 .eq. gym+gys) then !top side
               xt = top_v(top_i +i - xs + 1)
               xlt = top_v(top_i + i - xs)
            else
               xt = x_v(x_i + row + gxm)
            endif

            if ((i .gt. gxs) .and. (j+1 .lt. gys+gym)) then
               xlt = x_v(x_i + row - 1 + gxm)
            endif

            if ((i+1 .lt. gxs+gxm) .and. (j .gt. gys)) then
               xrb = x_v(x_i + row + 1 - gxm)
            endif

            d1 = (xc-xl)*rhx
            d2 = (xc-xr)*rhx
            d3 = (xc-xt)*rhy
            d4 = (xc-xb)*rhy
            d5 = (xrb-xr)*rhy
            d6 = (xrb-xb)*rhx
            d7 = (xlt-xl)*rhy
            d8 = (xlt-xt)*rhx

            f1 = sqrt(1.0 + d1*d1 + d7*d7)
            f2 = sqrt(1.0 + d1*d1 + d4*d4)
            f3 = sqrt(1.0 + d3*d3 + d8*d8)
            f4 = sqrt(1.0 + d3*d3 + d2*d2)
            f5 = sqrt(1.0 + d2*d2 + d5*d5)
            f6 = sqrt(1.0 + d4*d4 + d6*d6)

            hl = (-hydhx*(1.0+d7*d7)+d1*d7)/(f1*f1*f1)+(-hydhx*(1.0+d4*d4)+d1*d4)/(f2*f2*f2)

            hr = (-hydhx*(1.0+d5*d5)+d2*d5)/(f5*f5*f5)+(-hydhx*(1.0+d3*d3)+d2*d3)/(f4*f4*f4)

            ht = (-hxdhy*(1.0+d8*d8)+d3*d8)/(f3*f3*f3)+(-hxdhy*(1.0+d2*d2)+d2*d3)/(f4*f4*f4)

            hb = (-hxdhy*(1.0+d6*d6)+d4*d6)/(f6*f6*f6)+(-hxdhy*(1.0+d1*d1)+d1*d4)/(f2*f2*f2)

            hbr = -d2*d5/(f5*f5*f5) - d4*d6/(f6*f6*f6)
            htl = -d1*d7/(f1*f1*f1) - d3*d8/(f3*f3*f3)

            hc = hydhx*(1.0+d7*d7)/(f1*f1*f1) + hxdhy*(1.0+d8*d8)/(f3*f3*f3) + hydhx*(1.0+d5*d5)/(f5*f5*f5) +                      &
     &           hxdhy*(1.0+d6*d6)/(f6*f6*f6) + (hxdhy*(1.0+d1*d1)+hydhx*(1.0+d4*d4)- 2*d1*d4)/(f2*f2*f2) +  (hxdhy*(1.0+d2*d2)+   &
     &           hydhx*(1.0+d3*d3)-2*d2*d3)/(f4*f4*f4)

            hl = hl * 0.5
            hr = hr * 0.5
            ht = ht * 0.5
            hb = hb * 0.5
            hbr = hbr * 0.5
            htl = htl * 0.5
            hc = hc * 0.5

            k = 0

            if (j .gt. 0) then
               v(k) = hb
               col(k) = row - gxm
               k=k+1
            endif

            if ((j .gt. 0) .and. (i .lt. mx-1)) then
               v(k) = hbr
               col(k) = row-gxm+1
               k=k+1
            endif

            if (i .gt. 0) then
               v(k) = hl
               col(k) = row - 1
               k = k+1
            endif

            v(k) = hc
            col(k) = row
            k=k+1

            if (i .lt. mx-1) then
               v(k) = hr
               col(k) = row + 1
               k=k+1
            endif

            if ((i .gt. 0) .and. (j .lt. my-1)) then
               v(k) = htl
               col(k) = row + gxm - 1
               k=k+1
            endif

            if (j .lt. my-1) then
               v(k) = ht
               col(k) = row + gxm
               k=k+1
            endif

! Set matrix values using local numbering, defined earlier in main routine
            PetscCall(MatSetValuesLocal(Hessian,i1,row,k,col,v,INSERT_VALUES,ierr))

         enddo
      enddo

! restore vectors
      PetscCall(VecRestoreArray(localX,x_v,x_i,ierr))
      PetscCall(VecRestoreArray(Left,left_v,left_i,ierr))
      PetscCall(VecRestoreArray(Right,right_v,right_i,ierr))
      PetscCall(VecRestoreArray(Top,top_v,top_i,ierr))
      PetscCall(VecRestoreArray(Bottom,bottom_v,bottom_i,ierr))

! Assemble the matrix
      PetscCall(MatAssemblyBegin(Hessian,MAT_FINAL_ASSEMBLY,ierr))
      PetscCall(MatAssemblyEnd(Hessian,MAT_FINAL_ASSEMBLY,ierr))

      PetscCall(PetscLogFlops(199.0d0*xm*ym,ierr))

      return
      end

! Top,Left,Right,Bottom,bheight,mx,my,bmx,bmy,H, defined in plate2f.h

! ----------------------------------------------------------------------------
!
!/*
!     MSA_BoundaryConditions - calculates the boundary conditions for the region
!
!
!*/

      subroutine MSA_BoundaryConditions(ierr)
      use mymodule
      implicit none

      PetscErrorCode   ierr
      PetscInt         i,j,k,limit,maxits
      PetscInt         xs, xm, gxs, gxm
      PetscInt         ys, ym, gys, gym
      PetscInt         bsize, lsize
      PetscInt         tsize, rsize
      PetscReal      one,two,three,tol
      PetscReal      scl,fnorm,det,xt
      PetscReal      yt,hx,hy,u1,u2,nf1,nf2
      PetscReal      njac11,njac12,njac21,njac22
      PetscReal      b, t, l, r
      PetscReal      boundary_v(0:1)
      PetscOffset      boundary_i
      logical exitloop
      PetscBool flg

      limit=0
      maxits = 5
      tol=1e-10
      b=-0.5
      t= 0.5
      l=-0.5
      r= 0.5
      xt=0
      yt=0
      one=1.0
      two=2.0
      three=3.0

      PetscCall(DMDAGetCorners(dm,xs,ys,PETSC_NULL_INTEGER,xm,ym,PETSC_NULL_INTEGER,ierr))
      PetscCall(DMDAGetGhostCorners(dm,gxs,gys,PETSC_NULL_INTEGER,gxm,gym,PETSC_NULL_INTEGER,ierr))

      bsize = xm + 2
      lsize = ym + 2
      rsize = ym + 2
      tsize = xm + 2

      PetscCall(VecCreateMPI(PETSC_COMM_WORLD,bsize,PETSC_DECIDE,Bottom,ierr))
      PetscCall(VecCreateMPI(PETSC_COMM_WORLD,tsize,PETSC_DECIDE,Top,ierr))
      PetscCall(VecCreateMPI(PETSC_COMM_WORLD,lsize,PETSC_DECIDE,Left,ierr))
      PetscCall(VecCreateMPI(PETSC_COMM_WORLD,rsize,PETSC_DECIDE,Right,ierr))

      hx= (r-l)/(mx+1)
      hy= (t-b)/(my+1)

      do j=0,3

         if (j.eq.0) then
            yt=b
            xt=l+hx*xs
            limit=bsize
            PetscCall(VecGetArray(Bottom,boundary_v,boundary_i,ierr))

         elseif (j.eq.1) then
            yt=t
            xt=l+hx*xs
            limit=tsize
            PetscCall(VecGetArray(Top,boundary_v,boundary_i,ierr))

         elseif (j.eq.2) then
            yt=b+hy*ys
            xt=l
            limit=lsize
            PetscCall(VecGetArray(Left,boundary_v,boundary_i,ierr))

         elseif (j.eq.3) then
            yt=b+hy*ys
            xt=r
            limit=rsize
            PetscCall(VecGetArray(Right,boundary_v,boundary_i,ierr))
         endif

         do i=0,limit-1

            u1=xt
            u2=-yt
            k = 0
            exitloop = .false.
            do while (k .lt. maxits .and. (.not. exitloop))

               nf1=u1 + u1*u2*u2 - u1*u1*u1/three-xt
               nf2=-u2 - u1*u1*u2 + u2*u2*u2/three-yt
               fnorm=sqrt(nf1*nf1+nf2*nf2)
               if (fnorm .gt. tol) then
                  njac11=one+u2*u2-u1*u1
                  njac12=two*u1*u2
                  njac21=-two*u1*u2
                  njac22=-one - u1*u1 + u2*u2
                  det = njac11*njac22-njac21*njac12
                  u1 = u1-(njac22*nf1-njac12*nf2)/det
                  u2 = u2-(njac11*nf2-njac21*nf1)/det
               else
                  exitloop = .true.
               endif
               k=k+1
            enddo

            boundary_v(i + boundary_i) = u1*u1-u2*u2
            if ((j .eq. 0) .or. (j .eq. 1)) then
               xt = xt + hx
            else
               yt = yt + hy
            endif

         enddo

         if (j.eq.0) then
            PetscCall(VecRestoreArray(Bottom,boundary_v,boundary_i,ierr))
         elseif (j.eq.1) then
            PetscCall(VecRestoreArray(Top,boundary_v,boundary_i,ierr))
         elseif (j.eq.2) then
            PetscCall(VecRestoreArray(Left,boundary_v,boundary_i,ierr))
         elseif (j.eq.3) then
            PetscCall(VecRestoreArray(Right,boundary_v,boundary_i,ierr))
         endif

      enddo

! Scale the boundary if desired
      PetscCall(PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-bottom',scl,flg,ierr))
      if (flg .eqv. PETSC_TRUE) then
         PetscCall(VecScale(Bottom,scl,ierr))
      endif

      PetscCall(PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-top',scl,flg,ierr))
      if (flg .eqv. PETSC_TRUE) then
         PetscCall(VecScale(Top,scl,ierr))
      endif

      PetscCall(PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-right',scl,flg,ierr))
      if (flg .eqv. PETSC_TRUE) then
         PetscCall(VecScale(Right,scl,ierr))
      endif

      PetscCall(PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-left',scl,flg,ierr))
      if (flg .eqv. PETSC_TRUE) then
         PetscCall(VecScale(Left,scl,ierr))
      endif

      return
      end

! ----------------------------------------------------------------------------
!
!/*
!     MSA_Plate - Calculates an obstacle for surface to stretch over
!
!     Output Parameter:
!.    xl - lower bound vector
!.    xu - upper bound vector
!
!*/

      subroutine MSA_Plate(tao,xl,xu,dummy,ierr)
      use mymodule
      implicit none

      Tao        tao
      Vec              xl,xu
      PetscErrorCode   ierr
      PetscInt         i,j,row
      PetscInt         xs, xm, ys, ym
      PetscReal      lb,ub
      PetscInt         dummy

! PETSc's VecGetArray acts differently in Fortran than it does in C.
! Calling VecGetArray((Vec) X, (PetscReal) x_array(0:1), (PetscOffset) x_index, ierr)
! will return an array of doubles referenced by x_array offset by x_index.
!  i.e.,  to reference the kth element of X, use x_array(k + x_index).
! Notice that by declaring the arrays with range (0:1), we are using the C 0-indexing practice.
      PetscReal      xl_v(0:1)
      PetscOffset      xl_i

      lb = PETSC_NINFINITY
      ub = PETSC_INFINITY

      if (bmy .lt. 0) bmy = 0
      if (bmy .gt. my) bmy = my
      if (bmx .lt. 0) bmx = 0
      if (bmx .gt. mx) bmx = mx

      PetscCall(DMDAGetCorners(dm,xs,ys,PETSC_NULL_INTEGER,xm,ym,PETSC_NULL_INTEGER,ierr))

      PetscCall(VecSet(xl,lb,ierr))
      PetscCall(VecSet(xu,ub,ierr))

      PetscCall(VecGetArray(xl,xl_v,xl_i,ierr))

      do i=xs,xs+xm-1

         do j=ys,ys+ym-1

            row=(j-ys)*xm + (i-xs)

            if (i.ge.((mx-bmx)/2) .and. i.lt.(mx-(mx-bmx)/2) .and.           &
     &          j.ge.((my-bmy)/2) .and. j.lt.(my-(my-bmy)/2)) then
               xl_v(xl_i+row) = bheight

            endif

         enddo
      enddo

      PetscCall(VecRestoreArray(xl,xl_v,xl_i,ierr))

      return
      end

! ----------------------------------------------------------------------------
!
!/*
!     MSA_InitialPoint - Calculates an obstacle for surface to stretch over
!
!     Output Parameter:
!.    X - vector for initial guess
!
!*/

      subroutine MSA_InitialPoint(X, ierr)
      use mymodule
      implicit none

      Vec               X
      PetscErrorCode    ierr
      PetscInt          start,i,j
      PetscInt          row
      PetscInt          xs,xm,gxs,gxm
      PetscInt          ys,ym,gys,gym
      PetscReal       zero, np5

! PETSc's VecGetArray acts differently in Fortran than it does in C.
! Calling VecGetArray((Vec) X, (PetscReal) x_array(0:1), (integer) x_index, ierr)
! will return an array of doubles referenced by x_array offset by x_index.
!  i.e.,  to reference the kth element of X, use x_array(k + x_index).
! Notice that by declaring the arrays with range (0:1), we are using the C 0-indexing practice.
      PetscReal   left_v(0:1),right_v(0:1)
      PetscReal   bottom_v(0:1),top_v(0:1)
      PetscReal   x_v(0:1)
      PetscOffset   left_i, right_i, top_i
      PetscOffset   bottom_i,x_i
      PetscBool     flg
      PetscRandom   rctx

      zero = 0.0
      np5 = -0.5

      PetscCall(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-start', start,flg,ierr))

      if ((flg .eqv. PETSC_TRUE) .and. (start .eq. 0)) then  ! the zero vector is reasonable
         PetscCall(VecSet(X,zero,ierr))

      elseif ((flg .eqv. PETSC_TRUE) .and. (start .gt. 0)) then  ! random start -0.5 < xi < 0.5
         PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,rctx,ierr))
         do i=0,start-1
            PetscCall(VecSetRandom(X,rctx,ierr))
         enddo

         PetscCall(PetscRandomDestroy(rctx,ierr))
         PetscCall(VecShift(X,np5,ierr))

      else   ! average of boundary conditions

!        Get Local mesh boundaries
         PetscCall(DMDAGetCorners(dm,xs,ys,PETSC_NULL_INTEGER,xm,ym,PETSC_NULL_INTEGER,ierr))
         PetscCall(DMDAGetGhostCorners(dm,gxs,gys,PETSC_NULL_INTEGER,gxm,gym,PETSC_NULL_INTEGER,ierr))

!        Get pointers to vector data
         PetscCall(VecGetArray(Top,top_v,top_i,ierr))
         PetscCall(VecGetArray(Bottom,bottom_v,bottom_i,ierr))
         PetscCall(VecGetArray(Left,left_v,left_i,ierr))
         PetscCall(VecGetArray(Right,right_v,right_i,ierr))

         PetscCall(VecGetArray(localX,x_v,x_i,ierr))

!        Perform local computations
         do  j=ys,ys+ym-1
            do i=xs,xs+xm-1
               row = (j-gys)*gxm  + (i-gxs)
               x_v(x_i + row) = ((j+1)*bottom_v(bottom_i +i-xs+1)/my + (my-j+1)*top_v(top_i+i-xs+1)/(my+2) +                  &
     &                          (i+1)*left_v(left_i+j-ys+1)/mx + (mx-i+1)*right_v(right_i+j-ys+1)/(mx+2))*0.5
            enddo
         enddo

!        Restore vectors
         PetscCall(VecRestoreArray(localX,x_v,x_i,ierr))

         PetscCall(VecRestoreArray(Left,left_v,left_i,ierr))
         PetscCall(VecRestoreArray(Top,top_v,top_i,ierr))
         PetscCall(VecRestoreArray(Bottom,bottom_v,bottom_i,ierr))
         PetscCall(VecRestoreArray(Right,right_v,right_i,ierr))

         PetscCall(DMLocalToGlobalBegin(dm,localX,INSERT_VALUES,X,ierr))
         PetscCall(DMLocalToGlobalEnd(dm,localX,INSERT_VALUES,X,ierr))

      endif

      return
      end

!
!/*TEST
!
!   build:
!      requires: !complex
!
!   test:
!      args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bqnls -tao_gatol 1.e-4
!      filter: sort -b
!      filter_output: sort -b
!      requires: !single
!
!   test:
!      suffix: 2
!      nsize: 2
!      args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bqnls -tao_gatol 1.e-4
!      filter: sort -b
!      filter_output: sort -b
!      requires: !single
!
!TEST*/
