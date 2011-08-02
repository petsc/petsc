!  Program usage: mpirun -np <proc> eptorsion2f [all TAO options] 
!
!  Description:  This example demonstrates use of the TAO package to solve
!  unconstrained minimization problems in parallel.  This example is based 
!  on the Elastic-Plastic Torsion (dept) problem from the MINPACK-2 test suite.
!  The command line options are:
!    -mx <xg>, where <xg> = number of grid points in the 1st coordinate direction
!    -my <yg>, where <yg> = number of grid points in the 2nd coordinate direction
!    -par <param>, where <param> = angle of twist per unit length
!
!/*T
!   Concepts: TAO - Solving an unconstrained minimization problem
!   Routines: TaoInitialize(); TaoFinalize(); 
!   Routines: TaoCreate(); TaoDestroy();
!   Routines: TaoApplicationCreate(); TaoAppDestroy();
!   Routines: TaoAppSetObjectiveAndGradientRoutine();
!   Routines: TaoAppSetHessianMat(); TaoAppSetHessianRoutine();
!   Routines: TaoSetOptions();
!   Routines: TaoAppSetInitialSolutionVec(); TaoSolveApplication(); TaoDestroy();
!   Routines: TaoGetSolutionStatus();
!   Processors: n
!T*/
!
! ---------------------------------------------------------------------- 
!
!  Elastic-plastic torsion problem.  
!
!  The elastic plastic torsion problem arises from the determination 
!  of the stress field on an infinitely long cylindrical bar, which is
!  equivalent to the solution of the following problem:
!     min{ .5 * integral(||gradient(v(x))||^2 dx) - C * integral(v(x) dx)}
!  where C is the torsion angle per unit length.
!
!  The C version of this code is eptorsion2.c
!
! ---------------------------------------------------------------------- 

      implicit none
#include "eptorsion2f.h"

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!
!  See additional variable declarations in the file eptorsion2f.h
!
      PetscErrorCode   ierr           ! used to check for functions returning nonzeros
      Vec              x              ! solution vector
      Mat              H              ! hessian matrix
      PetscInt         Nx, Ny         ! number of processes in x- and y- directions
      TaoSolver        tao            ! TaoSolver solver context
      TaoSolverTerminationReason reason
      PetscBool        flg
      PetscInt         iter           ! iteration information
      PetscReal      ff,gnorm,cnorm,xdiff
      PetscInt         i1
      

!  Note: Any user-defined Fortran routines (such as FormGradient)
!  MUST be declared as external.

      external FormInitialGuess,FormFunctionGradient,ComputeHessian

      i1 = 1

!     Initialize TAO, PETSc  contexts
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call TaoInitialize(PETSC_NULL_CHARACTER,ierr)

!     Specify default parameters 
      param = 5.0d0
      mx = 10
      my = 10
      Nx = PETSC_DECIDE
      Ny = PETSC_DECIDE

!     Check for any command line arguments that might override defaults
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,"-mx",mx,flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,"-my",my,flg,ierr)
      call PetscOptionsGetReal(PETSC_NULL_CHARACTER,"-par",              &
     &                         param,flg,ierr)

      
!     Set up distributed array and vectors
      call DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,             &
     &     DMDA_BOUNDARY_NONE,                                           &
     &     DMDA_STENCIL_BOX,mx,my,Nx,Ny,i1,i1,PETSC_NULL_INTEGER,        &
     &     PETSC_NULL_INTEGER,dm,ierr)

!     Create vectors
      call DMCreateGlobalVector(dm,x,ierr)
      call DMCreateLocalVector(dm,localX,ierr)

!     Create Hessian
      call DMGetMatrix(dm,MATAIJ,H,ierr)
      call MatSetOption(H,MAT_SYMMETRIC,PETSC_TRUE,ierr)

!     The TAO code begins here

!     Create TAO solver
      call TaoSolverCreate(PETSC_COMM_WORLD,tao,ierr)
      call TaoSolverSetType(tao,"tao_cg",ierr)

!     Set routines for function and gradient evaluation

      call TaoSolverSetObjectiveAndGradientRoutine(tao,                  &
     &     FormFunctionGradient,PETSC_NULL_OBJECT,ierr)
      call TaoSolverSetHessianRoutine(tao,H,H,ComputeHessian,            &
     &     PETSC_NULL_OBJECT,ierr)

!     Set initial guess
      call FormInitialGuess(x,ierr)
      call TaoSolverSetInitialVector(tao,x,ierr)

!     Check for any TAO command line options
      call TaoSolverSetFromOptions(tao,ierr)


!     SOLVE THE APPLICATION
      call TaoSolverSolve(tao,ierr)

!     Get information on termination
      call TaoSolverGetTerminationReason(tao,reason,ierr)
      if (reason .lt. 0) then
          print *,'TAO did not terminate successfully'
      endif

      
!     Free TAO data structures
      call TaoSolverDestroy(tao,ierr)

  
!     Free PETSc data structures 
      call VecDestroy(x,ierr)
      call VecDestroy(localX,ierr)
      call MatDestroy(H,ierr)
      call DMDestroy(dm,ierr)


!     Finalize TAO and PETSc
      call PetscFinalize(ierr)
      call TaoFinalize(ierr)

      end


! ---------------------------------------------------------------------
!
!   FormInitialGuess - Computes an initial approximation to the solution.
!
!   Input Parameters:
!   X    - vector
!
!   Output Parameters:
!   X    - vector
!   ierr - error code
!
      subroutine FormInitialGuess(X,ierr)
      implicit none

! mx, my are defined in eptorsion2f.h
#include "eptorsion2f.h"

!  Input/output variables:
      Vec              X
      PetscErrorCode   ierr

!  Local variables:
      PetscInt         i, j, k, xe, ye
      PetscReal      temp, val, hx, hy
      PetscInt         xs, ys, xm, ym
      PetscInt         gxm, gym, gxs, gys
      PetscInt         i1

      i1 = 1
      hx = 1.0d0/(mx + 1)
      hy = 1.0d0/(my + 1)

!  Get corner information
      call DMDAGetCorners(dm,xs,ys,PETSC_NULL_INTEGER,xm,ym,               &
     &                  PETSC_NULL_INTEGER,ierr) 
      call DMDAGetGhostCorners(dm,gxs,gys,PETSC_NULL_INTEGER,              &
     &                   gxm,gym,PETSC_NULL_INTEGER,ierr)



!  Compute initial guess over locally owned part of mesh
      xe = xs+xm
      ye = ys+ym
      do j=ys,ye-1
         temp = min(j+1,my-j)*hy
         do i=xs,xe-1
            k   = (j-gys)*gxm + i-gxs
            val = min((min(i+1,mx-i))*hx,temp)
            call VecSetValuesLocal(X,i1,k,val,ADD_VALUES,ierr)
         end do
      end do
      call VecAssemblyBegin(X,ierr)
      call VecAssemblyEnd(X,ierr)
      return
      end


! ---------------------------------------------------------------------
!
!  FormFunctionGradient - Evaluates gradient G(X). 
!
!  Input Parameters:
!  tao   - the TaoSolver context
!  X     - input vector
!  dummy - optional user-defined context (not used here)
!    
!  Output Parameters:
!  f     - the function value at X
!  G     - vector containing the newly evaluated gradient
!  ierr  - error code
!
!  Notes:
!  This routine serves as a wrapper for the lower-level routine
!  "ApplicationGradient", where the actual computations are 
!  done using the standard Fortran style of treating the local
!  input vector data as an array over the local mesh.
!
      subroutine FormFunctionGradient(tao,X,f,G,dummy,ierr)
      implicit none

! dm, mx, my, param, localX declared in eptorsion2f.h
#include "eptorsion2f.h"

!  Input/output variables:
      TaoSolver        tao
      Vec              X, G
      PetscReal      f
      PetscErrorCode   ierr
      PetscInt         dummy

!  Declarations for use with local array:


! PETSc's VecGetArray acts differently in Fortran than it does in C.
! Calling VecGetArray((Vec) X, (PetscReal) x_array(0:1), (PetscOffset) x_index, ierr)
! will return an array of doubles referenced by x_array offset by x_index.
!  i.e.,  to reference the kth element of X, use x_array(k + x_index).
! Notice that by declaring the arrays with range (0:1), we are using the C 0-indexing practice.
      PetscReal      lx_v(0:1)
      PetscOffset      lx_i

!  Local variables:
      PetscReal      zero, p5, area, cdiv3
      PetscReal      val, flin, fquad,floc
      PetscReal      v, vb, vl, vr, vt, dvdx
      PetscReal      dvdy, hx, hy
      PetscInt         xe, ye, xsm, ysm
      PetscInt         xep, yep, i, j, k, ind
      PetscInt         xs, ys, xm, ym 
      PetscInt         gxs, gys, gxm, gym
      PetscInt         i1

      i1 = 1
      ierr  = 0
      cdiv3 = param/3.0d0
      zero = 0.0d0
      p5   = 0.5d0
      hx = 1.0d0/(mx + 1)
      hy = 1.0d0/(my + 1)
      fquad = zero
      flin = zero

!  Initialize gradient to zero
      call VecSet(G,zero,ierr)

!  Scatter ghost points to local vector
      call DMGlobalToLocalBegin(dm,X,INSERT_VALUES,localX,ierr)
      call DMGlobalToLocalEnd(dm,X,INSERT_VALUES,localX,ierr)


!  Get corner information
      call DMDAGetCorners(dm,xs,ys,PETSC_NULL_INTEGER,xm,ym,               &
     &                  PETSC_NULL_INTEGER,ierr)
      call DMDAGetGhostCorners(dm,gxs,gys,PETSC_NULL_INTEGER,              &
     &                   gxm,gym,PETSC_NULL_INTEGER,ierr)

!  Get pointer to vector data.
      call VecGetArray(localX,lx_v,lx_i,ierr)


!  Set local loop dimensions
      xe = xs+xm
      ye = ys+ym
      if (xs .eq. 0) then
         xsm = xs-1
      else
         xsm = xs
      endif
      if (ys .eq. 0) then
         ysm = ys-1
      else
         ysm = ys
      endif
      if (xe .eq. mx) then
         xep = xe+1
      else
         xep = xe
      endif
      if (ye .eq. my) then
         yep = ye+1
      else
         yep = ye
      endif

!     Compute local gradient contributions over the lower triangular elements
     
      do j = ysm, ye-1
         do i = xsm, xe-1
            k  = (j-gys)*gxm + i-gxs
            v  = zero
            vr = zero
            vt = zero
            if (i .ge. 0 .and. j .ge. 0)      v = lx_v(lx_i+k)
            if (i .lt. mx-1 .and. j .gt. -1) vr = lx_v(lx_i+k+1)
            if (i .gt. -1 .and. j .lt. my-1) vt = lx_v(lx_i+k+gxm)
            dvdx = (vr-v)/hx
            dvdy = (vt-v)/hy
            if (i .ne. -1 .and. j .ne. -1) then
               ind = k
               val = - dvdx/hx - dvdy/hy - cdiv3
               call VecSetValuesLocal(G,i1,k,val,ADD_VALUES,ierr)
            endif
            if (i .ne. mx-1 .and. j .ne. -1) then
               ind = k+1
               val =  dvdx/hx - cdiv3
               call VecSetValuesLocal(G,i1,ind,val,ADD_VALUES,ierr)
            endif
            if (i .ne. -1 .and. j .ne. my-1) then
              ind = k+gxm
              val = dvdy/hy - cdiv3
              call VecSetValuesLocal(G,i1,ind,val,ADD_VALUES,ierr)
            endif
            fquad = fquad + dvdx*dvdx + dvdy*dvdy
            flin = flin - cdiv3 * (v+vr+vt)
         end do
      end do

!     Compute local gradient contributions over the upper triangular elements

      do j = ys, yep-1
         do i = xs, xep-1
            k  = (j-gys)*gxm + i-gxs
            vb = zero
            vl = zero
            v  = zero
            if (i .lt. mx .and. j .gt. 0) vb = lx_v(lx_i+k-gxm)
            if (i .gt. 0 .and. j .lt. my) vl = lx_v(lx_i+k-1)
            if (i .lt. mx .and. j .lt. my) v = lx_v(lx_i+k)
            dvdx = (v-vl)/hx
            dvdy = (v-vb)/hy
            if (i .ne. mx .and. j .ne. 0) then
               ind = k-gxm
               val = - dvdy/hy - cdiv3
               call VecSetValuesLocal(G,i1,ind,val,ADD_VALUES,ierr)
            endif
            if (i .ne. 0 .and. j .ne. my) then
               ind = k-1
               val =  - dvdx/hx - cdiv3
               call VecSetValuesLocal(G,i1,ind,val,ADD_VALUES,ierr)
            endif
            if (i .ne. mx .and. j .ne. my) then
               ind = k
               val =  dvdx/hx + dvdy/hy - cdiv3
               call VecSetValuesLocal(G,i1,ind,val,ADD_VALUES,ierr)
            endif
            fquad = fquad + dvdx*dvdx + dvdy*dvdy
            flin = flin - cdiv3*(vb + vl + v)
         end do
      end do

!  Restore vector
      call VecRestoreArray(localX,lx_v,lx_i,ierr)

!  Assemble gradient vector
      call VecAssemblyBegin(G,ierr)
      call VecAssemblyEnd(G,ierr)

! Scale the gradient      
      area = p5*hx*hy
      floc = area *(p5*fquad+flin)
      call VecScale(G,area,ierr)


!  Sum function contributions from all processes
      call MPI_Allreduce(floc,f,1,MPIU_SCALAR,MPI_SUM,                      &
     &                   PETSC_COMM_WORLD,ierr)
      call PetscLogFlops((ye-ysm)*(xe-xsm)*20+(xep-xs)*(yep-ys)*16.0d0,     &
     &                    ierr)



      return
      end




      subroutine ComputeHessian(tao, X, H, Hpre, flag, dummy, ierr)
      implicit none
#include "eptorsion2f.h"      
      TaoSolver       tao
      Vec             X
      Mat             H,Hpre
      MatStructure    flag
      PetscErrorCode  ierr
      PetscInt        dummy

      
      PetscInt i,j,k
      PetscInt col(0:4),row
      PetscInt xs,xm,gxs,gxm
      PetscInt ys,ym,gys,gym
      PetscReal v(0:4)
      PetscInt i1

      i1 = 1

!     Get local grid boundaries
      call DMDAGetCorners(dm,xs,ys,PETSC_NULL_INTEGER,xm,ym,               &
     &                PETSC_NULL_INTEGER,ierr)
      call DMDAGetGhostCorners(dm,gxs,gys,PETSC_NULL_INTEGER,gxm,gym,      &
     &                PETSC_NULL_INTEGER,ierr)

      do j=ys,ys+ym-1
         do i=xs,xs+xm-1
            row = (j-gys)*gxm + (i-gxs)

            k = 0
            if (j .gt. gys) then
               v(k) = -1.0d0
               col(k) = row-gxm
               k = k + 1
            endif

            if (i .gt. gxs) then
               v(k) = -1.0d0
               col(k) = row - 1
               k = k +1
            endif

            v(k) = 4.0d0
            col(k) = row
            k = k + 1

            if (i+1 .lt. gxs + gxm) then
               v(k) = -1.0d0
               col(k) = row + 1
               k = k + 1
            endif

            if (j+1 .lt. gys + gym) then
               v(k) = -1.0d0
               col(k) = row + gxm
               k = k + 1
            endif

            call MatSetValuesLocal(H,i1,row,k,col,v,INSERT_VALUES,ierr)
         enddo
      enddo

      
!     Assemble matrix
      call MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY,ierr)


!     Tell the matrix we will never add a new nonzero location to the
!     matrix.  If we do it will generate an error.

      call MatSetOption(H,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE,ierr)
      call MatSetOption(H,MAT_SYMMETRIC,PETSC_TRUE,ierr)


      call PetscLogFlops(9.0d0*xm*ym + 49.0d0*xm,ierr)

      ierr = 0
      return 
      end
      
               
      







