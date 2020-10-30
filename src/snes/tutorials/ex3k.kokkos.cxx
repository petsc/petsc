static char help[] = "Newton methods to solve u'' + u^{2} = f in parallel. Uses Kokkos\n\\n";

#include <petscdmda_kokkos.hpp>
#include <petscsnes.h>

/*
   User-defined application context
*/
typedef struct {
  DM          da;      /* distributed array */
  Vec         F;       /* right-hand-side of PDE */
  PetscReal   h;       /* mesh spacing */
} ApplicationCtx;

/* ------------------------------------------------------------------- */
/*
   FormInitialGuess - Computes initial guess.

   Input/Output Parameter:
.  x - the solution vector
*/
PetscErrorCode FormInitialGuess(Vec x)
{
  PetscErrorCode ierr;
  PetscScalar    pfive = .50;

  PetscFunctionBeginUser;
  ierr = VecSet(x,pfive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
   CpuFunction - Evaluates nonlinear function, F(x) on CPU

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  ctx - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  r - function vector

   Note:
   The user-defined context can contain any application-specific
   data needed for the function evaluation.
*/
PetscErrorCode CpuFunction(SNES snes,Vec x,Vec r,void *ctx)
{
  ApplicationCtx *user = (ApplicationCtx*) ctx;
  DM             da    = user->da;
  PetscScalar    *X,*R,*F,d;
  PetscErrorCode ierr;
  PetscInt       i,M,xs,xm;
  Vec            xl;

  PetscFunctionBeginUser;
  ierr = DMGetLocalVector(da,&xl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(da,x,INSERT_VALUES,xl);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,xl,&X);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,r,&R);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,user->F,&F);CHKERRQ(ierr);

  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  if (xs == 0) { /* left boundary */
    R[0] = X[0];
    xs++;xm--;
  }
  if (xs+xm == M) {  /* right boundary */
    R[xs+xm-1] = X[xs+xm-1] - 1.0;
    xm--;
  }
  d = 1.0/(user->h*user->h);
  for (i=xs; i<xs+xm; i++) R[i] = d*(X[i-1] - 2.0*X[i] + X[i+1]) + X[i]*X[i] - F[i];

  ierr = DMDAVecRestoreArray(da,xl,&X);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,r,&R);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,user->F,&F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&xl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

using DefaultExecutionSpace             = Kokkos::DefaultExecutionSpace;
using DefaultMemorySpace                = Kokkos::DefaultExecutionSpace::memory_space;
using PetscScalarKokkosOffsetView       = Kokkos::Experimental::OffsetView<PetscScalar*,DefaultMemorySpace>;
using ConstPetscScalarKokkosOffsetView  = Kokkos::Experimental::OffsetView<const PetscScalar*,DefaultMemorySpace>;

PetscErrorCode KokkosFunction(SNES snes,Vec x,Vec r,void *ctx)
{
  PetscErrorCode                       ierr;
  ApplicationCtx                       *user = (ApplicationCtx*) ctx;
  DM                                   da = user->da;
  PetscScalar                          d;
  PetscInt                             M;
  Vec                                  xl;
  PetscScalarKokkosOffsetView          R;
  ConstPetscScalarKokkosOffsetView     X,F;

  PetscFunctionBeginUser;
  ierr = DMGetLocalVector(da,&xl);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(da,x,INSERT_VALUES,xl);CHKERRQ(ierr);
  d    = 1.0/(user->h*user->h);
  ierr = DMDAGetInfo(da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAVecGetKokkosOffsetView(da,xl,&X);CHKERRQ(ierr); /* read only */
  ierr = DMDAVecGetKokkosOffsetViewWrite(da,r,&R);CHKERRQ(ierr); /* write only */
  ierr = DMDAVecGetKokkosOffsetView(da,user->F,&F);CHKERRQ(ierr); /* read only */
  Kokkos:: parallel_for (Kokkos::RangePolicy<>(R.begin(0),R.end(0)),KOKKOS_LAMBDA (int i) {
    if (i == 0)        R(0) = X(0);        /* left boundary */
    else if (i == M-1) R(i) = X(i) - 1.0;  /* right boundary */
    else               R(i) = d*(X(i-1) - 2.0*X(i) + X(i+1)) + X(i)*X(i) - F(i); /* interior */
  });
  ierr = DMDAVecRestoreKokkosOffsetView(da,xl,&X);CHKERRQ(ierr);
  ierr = DMDAVecRestoreKokkosOffsetViewWrite(da,r,&R);CHKERRQ(ierr);
  ierr = DMDAVecRestoreKokkosOffsetView(da,user->F,&F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&xl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StubFunction(SNES snes ,Vec x,Vec r,void *ctx)
{
  PetscErrorCode                       ierr;
  ApplicationCtx                       *user = (ApplicationCtx*) ctx;
  DM                                   da = user->da;
  Vec                                  rk;
  PetscReal                            norm=0;

  PetscFunctionBeginUser;
  ierr = DMGetGlobalVector(da,&rk);CHKERRQ(ierr);
  ierr = CpuFunction(snes,x,r,ctx);CHKERRQ(ierr);
  ierr = KokkosFunction(snes,x,rk,ctx);CHKERRQ(ierr);
  ierr = VecAXPY(rk,-1.0,r);CHKERRQ(ierr);
  ierr = VecNorm(rk,NORM_2,&norm);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&rk);CHKERRQ(ierr);
  if (norm > 1e-6) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"KokkosFunction() different from CpuFunction() with a diff norm = %g\n",norm);
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
/*
   FormJacobian - Evaluates Jacobian matrix.

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  dummy - optional user-defined context (not used here)

   Output Parameters:
.  jac - Jacobian matrix
.  B - optionally different preconditioning matrix
.  flag - flag indicating matrix structure
*/
PetscErrorCode FormJacobian(SNES snes,Vec x,Mat jac,Mat B,void *ctx)
{
  ApplicationCtx *user = (ApplicationCtx*) ctx;
  PetscScalar    *xx,d,A[3];
  PetscErrorCode ierr;
  PetscInt       i,j[3],M,xs,xm;
  DM             da = user->da;

  PetscFunctionBeginUser;
  /*
     Get pointer to vector data
  */
  ierr = DMDAVecGetArrayRead(da,x,&xx);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  /*
    Get range of locally owned matrix
  */
  ierr = DMDAGetInfo(da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  /*
     Determine starting and ending local indices for interior grid points.
     Set Jacobian entries for boundary points.
  */

  if (xs == 0) {  /* left boundary */
    i = 0; A[0] = 1.0;

    ierr = MatSetValues(jac,1,&i,1,&i,A,INSERT_VALUES);CHKERRQ(ierr);
    xs++;xm--;
  }
  if (xs+xm == M) { /* right boundary */
    i    = M-1;
    A[0] = 1.0;
    ierr = MatSetValues(jac,1,&i,1,&i,A,INSERT_VALUES);CHKERRQ(ierr);
    xm--;
  }

  /*
     Interior grid points
      - Note that in this case we set all elements for a particular
        row at once.
  */
  d = 1.0/(user->h*user->h);
  for (i=xs; i<xs+xm; i++) {
    j[0] = i - 1; j[1] = i; j[2] = i + 1;
    A[0] = A[2] = d; A[1] = -2.0*d + 2.0*xx[i];
    ierr = MatSetValues(jac,1,&i,3,j,A,INSERT_VALUES);CHKERRQ(ierr);
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.

     Also, restore vector.
  */

  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,x,&xx);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  SNES                        snes;                 /* SNES context */
  Mat                         J;                    /* Jacobian matrix */
  ApplicationCtx              ctx;                  /* user-defined context */
  Vec                         x,r,U,F;              /* vectors */
  PetscScalar                 none = -1.0;
  PetscErrorCode              ierr;
  PetscInt                    its,N = 5,maxit,maxf;
  PetscReal                   abstol,rtol,stol,norm;
  PetscBool                   viewinitial = PETSC_FALSE;
  PetscScalarKokkosOffsetView FF,UU;

  ierr  = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr  = PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL);CHKERRQ(ierr);
  ctx.h = 1.0/(N-1);
  ierr  = PetscOptionsGetBool(NULL,NULL,"-view_initial",&viewinitial,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
  */
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,N,1,1,NULL,&ctx.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(ctx.da);CHKERRQ(ierr);
  ierr = DMSetUp(ctx.da);CHKERRQ(ierr);

  /*
     Extract global and local vectors from DMDA; then duplicate for remaining
     vectors that are the same types
  */
  ierr = DMCreateGlobalVector(ctx.da,&x);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)x,"Approximate Solution");CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&F);CHKERRQ(ierr); ctx.F = F;
  ierr = PetscObjectSetName((PetscObject)F,"Forcing function");CHKERRQ(ierr);
  ierr = VecDuplicate(x,&U);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)U,"Exact Solution");CHKERRQ(ierr);

  /*
     Set function evaluation routine and vector.  Whenever the nonlinear
     solver needs to compute the nonlinear function, it will call this
     routine.
      - Note that the final routine argument is the user-defined
        context that provides application-specific data for the
        function evaluation routine.

     At the begining, one can use a stub function that checks the Kokkos version
     against the CPU version to quickly expose errors.
     ierr = SNESSetFunction(snes,r,StubFunction,&ctx);CHKERRQ(ierr);
  */
  ierr = SNESSetFunction(snes,r,KokkosFunction,&ctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateMatrix(ctx.da,&J);CHKERRQ(ierr);

  /*
     Set Jacobian matrix data structure and default Jacobian evaluation
     routine.  Whenever the nonlinear solver needs to compute the
     Jacobian matrix, it will call this routine.
      - Note that the final routine argument is the user-defined
        context that provides application-specific data for the
        Jacobian evaluation routine.
  */
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,&ctx);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESGetTolerances(snes,&abstol,&rtol,&stol,&maxit,&maxf);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"atol=%g, rtol=%g, stol=%g, maxit=%D, maxf=%D\n",(double)abstol,(double)rtol,(double)stol,maxit,maxf);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize application:
     Store forcing function of PDE and exact solution
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDAVecGetKokkosOffsetViewWrite(ctx.da,F,&FF);CHKERRQ(ierr);
  ierr = DMDAVecGetKokkosOffsetViewWrite(ctx.da,U,&UU);CHKERRQ(ierr);
  Kokkos:: parallel_for (Kokkos::RangePolicy<>(FF.begin(0),FF.end(0)),KOKKOS_LAMBDA (int i) {
    PetscReal xp = i*ctx.h;
    FF(i) = 6.0*xp + pow(xp+1.e-12,6.0); /* +1.e-12 is to prevent 0^6 */
    UU(i) = xp*xp*xp;
  });
  ierr = DMDAVecRestoreKokkosOffsetViewWrite(ctx.da,F,&FF);CHKERRQ(ierr);
  ierr = DMDAVecRestoreKokkosOffsetViewWrite(ctx.da,U,&UU);CHKERRQ(ierr);

  if (viewinitial) {
    ierr = VecView(U,NULL);CHKERRQ(ierr);
    ierr = VecView(F,NULL);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */
  ierr = FormInitialGuess(x);CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of SNES iterations = %D\n",its);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Check solution and clean up
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Check the error
  */
  ierr = VecAXPY(x,none,U);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g Iterations %D\n",(double)norm,its);CHKERRQ(ierr);

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&ctx.da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: kokkos_kernels

   test:
     requires: kokkos_kernels !complex !single cuda
     nsize: 2
     args: -dm_vec_type kokkos -dm_mat_type aijkokkos -view_initial -snes_monitor
     output_file: output/ex3k_1.out

TEST*/