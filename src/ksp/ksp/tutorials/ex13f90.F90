      module ex13f90module
#include <petsc/finclude/petscksp.h>
        use petscksp
        type User
          Vec x
          Vec b
          Mat A
          KSP ksp
          PetscInt m
          PetscInt n
        end type User
      end module

      program main
      use ex13f90module
      implicit none

!    User-defined context that contains all the data structures used
!    in the linear solution process.

!   Vec    x,b      /* solution vector, right-hand side vector and work vector */
!   Mat    A        /* sparse matrix */
!   KSP   ksp     /* linear solver context */
!   int    m,n      /* grid dimensions */
!
!   Since we cannot store Scalars and integers in the same context,
!   we store the integers/pointers in the user-defined context, and
!   the scalar values are carried in the common block.
!   The scalar values in this simplistic example could easily
!   be recalculated in each routine, where they are needed.
!
!   Scalar hx2,hy2  /* 1/(m+1)*(m+1) and 1/(n+1)*(n+1) */

!  Note: Any user-defined Fortran routines MUST be declared as external.

      external UserInitializeLinearSolver
      external UserFinalizeLinearSolver
      external UserDoLinearSolver

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      PetscScalar  hx,hy,x,y
      type(User) userctx
      PetscErrorCode ierr
      PetscInt m,n,t,tmax,i,j
      PetscBool  flg
      PetscMPIInt size
      PetscReal  enorm
      PetscScalar cnorm
      PetscScalar,ALLOCATABLE :: userx(:,:)
      PetscScalar,ALLOCATABLE :: userb(:,:)
      PetscScalar,ALLOCATABLE :: solution(:,:)
      PetscScalar,ALLOCATABLE :: rho(:,:)

      PetscReal hx2,hy2
      common /param/ hx2,hy2

      tmax = 2
      m = 6
      n = 7

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      PetscCallA(PetscInitialize(ierr))
      PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD,size,ierr))
      PetscCheckA(size .eq. 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,'This is a uniprocessor example only')

!  The next two lines are for testing only; these allow the user to
!  decide the grid size at runtime.

      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-m',m,flg,ierr))
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',n,flg,ierr))

!  Create the empty sparse matrix and linear solver data structures

      PetscCallA(UserInitializeLinearSolver(m,n,userctx,ierr))

!  Allocate arrays to hold the solution to the linear system.  This
!  approach is not normally done in PETSc programs, but in this case,
!  since we are calling these routines from a non-PETSc program, we
!  would like to reuse the data structures from another code. So in
!  the context of a larger application these would be provided by
!  other (non-PETSc) parts of the application code.

      ALLOCATE (userx(m,n),userb(m,n),solution(m,n))

!  Allocate an array to hold the coefficients in the elliptic operator

       ALLOCATE (rho(m,n))

!  Fill up the array rho[] with the function rho(x,y) = x; fill the
!  right-hand side b and the solution with a known problem for testing.

      hx = 1.0/real(m+1)
      hy = 1.0/real(n+1)
      y  = hy
      do 20 j=1,n
         x = hx
         do 10 i=1,m
            rho(i,j)      = x
            solution(i,j) = sin(2.*PETSC_PI*x)*sin(2.*PETSC_PI*y)
            userb(i,j)    = -2.*PETSC_PI*cos(2.*PETSC_PI*x)*sin(2.*PETSC_PI*y) + 8*PETSC_PI*PETSC_PI*x*sin(2.*PETSC_PI*x)*sin(2.*PETSC_PI*y)
           x = x + hx
 10      continue
         y = y + hy
 20   continue

!  Loop over a bunch of timesteps, setting up and solver the linear
!  system for each time-step.
!  Note that this loop is somewhat artificial. It is intended to
!  demonstrate how one may reuse the linear solvers in each time-step.

      do 100 t=1,tmax
         PetscCallA(UserDoLinearSolver(rho,userctx,userb,userx,ierr))

!        Compute error: Note that this could (and usually should) all be done
!        using the PETSc vector operations. Here we demonstrate using more
!        standard programming practices to show how they may be mixed with
!        PETSc.
         cnorm = 0.0
         do 90 j=1,n
            do 80 i=1,m
              cnorm = cnorm + PetscConj(solution(i,j)-userx(i,j))*(solution(i,j)-userx(i,j))
 80         continue
 90      continue
         enorm =  PetscRealPart(cnorm*hx*hy)
         write(6,115) m,n,enorm
 115     format ('m = ',I2,' n = ',I2,' error norm = ',1PE11.4)
 100  continue

!  We are finished solving linear systems, so we clean up the
!  data structures.

      DEALLOCATE (userx,userb,solution,rho)

      PetscCallA(UserFinalizeLinearSolver(userctx,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

! ----------------------------------------------------------------
      subroutine UserInitializeLinearSolver(m,n,userctx,ierr)
      use ex13f90module
      implicit none

      PetscInt m,n
      PetscErrorCode ierr
      type(User) userctx

      common /param/ hx2,hy2
      PetscReal hx2,hy2

!  Local variable declararions
      Mat     A
      Vec     b,x
      KSP    ksp
      PetscInt Ntot,five,one

!  Here we assume use of a grid of size m x n, with all points on the
!  interior of the domain, i.e., we do not include the points corresponding
!  to homogeneous Dirichlet boundary conditions.  We assume that the domain
!  is [0,1]x[0,1].

      hx2 = (m+1)*(m+1)
      hy2 = (n+1)*(n+1)
      Ntot = m*n

      five = 5
      one = 1

!  Create the sparse matrix. Preallocate 5 nonzeros per row.

      PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,Ntot,Ntot,five,PETSC_NULL_INTEGER_ARRAY,A,ierr))
!
!  Create vectors. Here we create vectors with no memory allocated.
!  This way, we can use the data structures already in the program
!  by using VecPlaceArray() subroutine at a later stage.
!
      PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,one,Ntot,PETSC_NULL_SCALAR_ARRAY,b,ierr))
      PetscCall(VecDuplicate(b,x,ierr))

!  Create linear solver context. This will be used repeatedly for all
!  the linear solves needed.

      PetscCall(KSPCreate(PETSC_COMM_SELF,ksp,ierr))

      userctx%x = x
      userctx%b = b
      userctx%A = A
      userctx%ksp = ksp
      userctx%m = m
      userctx%n = n

      end
! -----------------------------------------------------------------------

!   Solves -div (rho grad psi) = F using finite differences.
!   rho is a 2-dimensional array of size m by n, stored in Fortran
!   style by columns. userb is a standard one-dimensional array.

      subroutine UserDoLinearSolver(rho,userctx,userb,userx,ierr)
      use ex13f90module
      implicit none

      PetscErrorCode ierr
      type(User) userctx
      PetscScalar rho(*),userb(*),userx(*)

      common /param/ hx2,hy2
      PetscReal hx2,hy2

      PC   pc
      KSP ksp
      Vec  b,x
      Mat  A
      PetscInt m,n,one
      PetscInt i,j,II,JJ
      PetscScalar  v

      one  = 1
      x    = userctx%x
      b    = userctx%b
      A    = userctx%A
      ksp  = userctx%ksp
      m    = userctx%m
      n    = userctx%n

!  This is not the most efficient way of generating the matrix,
!  but let's not worry about it.  We should have separate code for
!  the four corners, each edge and then the interior. Then we won't
!  have the slow if-tests inside the loop.
!
!  Compute the operator
!          -div rho grad
!  on an m by n grid with zero Dirichlet boundary conditions. The rho
!  is assumed to be given on the same grid as the finite difference
!  stencil is applied.  For a staggered grid, one would have to change
!  things slightly.

      II = 0
      do 110 j=1,n
         do 100 i=1,m
            if (j .gt. 1) then
               JJ = II - m
               v = -0.5*(rho(II+1) + rho(JJ+1))*hy2
               PetscCall(MatSetValues(A,one,[II],one,[JJ],[v],INSERT_VALUES,ierr))
            endif
            if (j .lt. n) then
               JJ = II + m
               v = -0.5*(rho(II+1) + rho(JJ+1))*hy2
               PetscCall(MatSetValues(A,one,[II],one,[JJ],[v],INSERT_VALUES,ierr))
            endif
            if (i .gt. 1) then
               JJ = II - 1
               v = -0.5*(rho(II+1) + rho(JJ+1))*hx2
               PetscCall(MatSetValues(A,one,[II],one,[JJ],[v],INSERT_VALUES,ierr))
            endif
            if (i .lt. m) then
               JJ = II + 1
               v = -0.5*(rho(II+1) + rho(JJ+1))*hx2
               PetscCall(MatSetValues(A,one,[II],one,[JJ],[v],INSERT_VALUES,ierr))
            endif
            v = 2*rho(II+1)*(hx2+hy2)
            PetscCall(MatSetValues(A,one,[II],one,[II],[v],INSERT_VALUES,ierr))
            II = II+1
 100     continue
 110  continue
!
!     Assemble matrix
!
      PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr))
      PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr))

!
!     Set operators. Here the matrix that defines the linear system
!     also serves as the matrix from which the preconditioner is constructed. Since all the matrices
!     will have the same nonzero pattern here, we indicate this so the
!     linear solvers can take advantage of this.
!
      PetscCall(KSPSetOperators(ksp,A,A,ierr))

!
!     Set linear solver defaults for this problem (optional).
!     - Here we set it to use direct LU factorization for the solution
!
      PetscCall(KSPGetPC(ksp,pc,ierr))
      PetscCall(PCSetType(pc,PCLU,ierr))

!
!     Set runtime options, e.g.,
!        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
!     These options will override those specified above as long as
!     KSPSetFromOptions() is called _after_ any other customization
!     routines.
!
!     Run the program with the option -help to see all the possible
!     linear solver options.
!
      PetscCall(KSPSetFromOptions(ksp,ierr))

!
!     This allows the PETSc linear solvers to compute the solution
!     directly in the user's array rather than in the PETSc vector.
!
!     This is essentially a hack and not highly recommend unless you
!     are quite comfortable with using PETSc. In general, users should
!     write their entire application using PETSc vectors rather than
!     arrays.
!
      PetscCall(VecPlaceArray(x,userx,ierr))
      PetscCall(VecPlaceArray(b,userb,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                      Solve the linear system
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      PetscCall(KSPSolve(ksp,b,x,ierr))

      PetscCall(VecResetArray(x,ierr))
      PetscCall(VecResetArray(b,ierr))
      end

! ------------------------------------------------------------------------

      subroutine UserFinalizeLinearSolver(userctx,ierr)
      use ex13f90module
      implicit none

      PetscErrorCode ierr
      type(User) userctx

!
!     We are all done and don't need to solve any more linear systems, so
!     we free the work space.  All PETSc objects should be destroyed when
!     they are no longer needed.
!
      PetscCall(VecDestroy(userctx%x,ierr))
      PetscCall(VecDestroy(userctx%b,ierr))
      PetscCall(MatDestroy(userctx%A,ierr))
      PetscCall(KSPDestroy(userctx%ksp,ierr))
      end

!
!/*TEST
!
!   test:
!      args: -m 19 -n 20 -ksp_gmres_cgs_refinement_type refine_always
!      output_file: output/ex13f90_1.out
!
!TEST*/
