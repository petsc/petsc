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
  PetscScalar    pfive = .50;

  PetscFunctionBeginUser;
  CHKERRQ(VecSet(x,pfive));
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
  PetscInt       i,M,xs,xm;
  Vec            xl;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetLocalVector(da,&xl));
  CHKERRQ(DMGlobalToLocal(da,x,INSERT_VALUES,xl));
  CHKERRQ(DMDAVecGetArray(da,xl,&X));
  CHKERRQ(DMDAVecGetArray(da,r,&R));
  CHKERRQ(DMDAVecGetArray(da,user->F,&F));

  CHKERRQ(DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL));
  CHKERRQ(DMDAGetInfo(da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));

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

  CHKERRQ(DMDAVecRestoreArray(da,xl,&X));
  CHKERRQ(DMDAVecRestoreArray(da,r,&R));
  CHKERRQ(DMDAVecRestoreArray(da,user->F,&F));
  CHKERRQ(DMRestoreLocalVector(da,&xl));
  PetscFunctionReturn(0);
}

using DefaultExecutionSpace             = Kokkos::DefaultExecutionSpace;
using DefaultMemorySpace                = Kokkos::DefaultExecutionSpace::memory_space;
using PetscScalarKokkosOffsetView       = Kokkos::Experimental::OffsetView<PetscScalar*,DefaultMemorySpace>;
using ConstPetscScalarKokkosOffsetView  = Kokkos::Experimental::OffsetView<const PetscScalar*,DefaultMemorySpace>;

PetscErrorCode KokkosFunction(SNES snes,Vec x,Vec r,void *ctx)
{
  ApplicationCtx                       *user = (ApplicationCtx*) ctx;
  DM                                   da = user->da;
  PetscScalar                          d;
  PetscInt                             M;
  Vec                                  xl;
  PetscScalarKokkosOffsetView          R;
  ConstPetscScalarKokkosOffsetView     X,F;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetLocalVector(da,&xl));
  CHKERRQ(DMGlobalToLocal(da,x,INSERT_VALUES,xl));
  d    = 1.0/(user->h*user->h);
  CHKERRQ(DMDAGetInfo(da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
  CHKERRQ(DMDAVecGetKokkosOffsetView(da,xl,&X)); /* read only */
  CHKERRQ(DMDAVecGetKokkosOffsetViewWrite(da,r,&R)); /* write only */
  CHKERRQ(DMDAVecGetKokkosOffsetView(da,user->F,&F)); /* read only */
  Kokkos:: parallel_for (Kokkos::RangePolicy<>(R.begin(0),R.end(0)),KOKKOS_LAMBDA (int i) {
    if (i == 0)        R(0) = X(0);        /* left boundary */
    else if (i == M-1) R(i) = X(i) - 1.0;  /* right boundary */
    else               R(i) = d*(X(i-1) - 2.0*X(i) + X(i+1)) + X(i)*X(i) - F(i); /* interior */
  });
  CHKERRQ(DMDAVecRestoreKokkosOffsetView(da,xl,&X));
  CHKERRQ(DMDAVecRestoreKokkosOffsetViewWrite(da,r,&R));
  CHKERRQ(DMDAVecRestoreKokkosOffsetView(da,user->F,&F));
  CHKERRQ(DMRestoreLocalVector(da,&xl));
  PetscFunctionReturn(0);
}

PetscErrorCode StubFunction(SNES snes ,Vec x,Vec r,void *ctx)
{
  ApplicationCtx                       *user = (ApplicationCtx*) ctx;
  DM                                   da = user->da;
  Vec                                  rk;
  PetscReal                            norm=0;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetGlobalVector(da,&rk));
  CHKERRQ(CpuFunction(snes,x,r,ctx));
  CHKERRQ(KokkosFunction(snes,x,rk,ctx));
  CHKERRQ(VecAXPY(rk,-1.0,r));
  CHKERRQ(VecNorm(rk,NORM_2,&norm));
  CHKERRQ(DMRestoreGlobalVector(da,&rk));
  PetscCheckFalse(norm > 1e-6,PETSC_COMM_SELF,PETSC_ERR_PLIB,"KokkosFunction() different from CpuFunction() with a diff norm = %g",norm);
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
  PetscInt       i,j[3],M,xs,xm;
  DM             da = user->da;

  PetscFunctionBeginUser;
  /*
     Get pointer to vector data
  */
  CHKERRQ(DMDAVecGetArrayRead(da,x,&xx));
  CHKERRQ(DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL));

  /*
    Get range of locally owned matrix
  */
  CHKERRQ(DMDAGetInfo(da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));

  /*
     Determine starting and ending local indices for interior grid points.
     Set Jacobian entries for boundary points.
  */

  if (xs == 0) {  /* left boundary */
    i = 0; A[0] = 1.0;

    CHKERRQ(MatSetValues(jac,1,&i,1,&i,A,INSERT_VALUES));
    xs++;xm--;
  }
  if (xs+xm == M) { /* right boundary */
    i    = M-1;
    A[0] = 1.0;
    CHKERRQ(MatSetValues(jac,1,&i,1,&i,A,INSERT_VALUES));
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
    CHKERRQ(MatSetValues(jac,1,&i,3,j,A,INSERT_VALUES));
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.

     Also, restore vector.
  */

  CHKERRQ(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  CHKERRQ(DMDAVecRestoreArrayRead(da,x,&xx));
  CHKERRQ(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));

  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  SNES           snes;          /* SNES context */
  Mat            J;             /* Jacobian matrix */
  ApplicationCtx ctx;           /* user-defined context */
  Vec            x,r,U,F;       /* vectors */
  PetscScalar    none        = -1.0;
  PetscInt       its,N       = 5,maxit,maxf;
  PetscReal      abstol,rtol,stol,norm;
  PetscBool      viewinitial = PETSC_FALSE;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL));
  ctx.h = 1.0/(N-1);
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-view_initial",&viewinitial,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(SNESCreate(PETSC_COMM_WORLD,&snes));

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
  */
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,N,1,1,NULL,&ctx.da));
  CHKERRQ(DMSetFromOptions(ctx.da));
  CHKERRQ(DMSetUp(ctx.da));

  /*
     Extract global and local vectors from DMDA; then duplicate for remaining
     vectors that are the same types
  */
  CHKERRQ(DMCreateGlobalVector(ctx.da,&x));
  CHKERRQ(PetscObjectSetName((PetscObject)x,"Approximate Solution"));
  CHKERRQ(VecDuplicate(x,&r));
  CHKERRQ(VecDuplicate(x,&F)); ctx.F = F;
  CHKERRQ(PetscObjectSetName((PetscObject)F,"Forcing function"));
  CHKERRQ(VecDuplicate(x,&U));
  CHKERRQ(PetscObjectSetName((PetscObject)U,"Exact Solution"));

  /*
     Set function evaluation routine and vector.  Whenever the nonlinear
     solver needs to compute the nonlinear function, it will call this
     routine.
      - Note that the final routine argument is the user-defined
        context that provides application-specific data for the
        function evaluation routine.

     At the beginning, one can use a stub function that checks the Kokkos version
     against the CPU version to quickly expose errors.
     CHKERRQ(SNESSetFunction(snes,r,StubFunction,&ctx));
  */
  CHKERRQ(SNESSetFunction(snes,r,KokkosFunction,&ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(DMCreateMatrix(ctx.da,&J));

  /*
     Set Jacobian matrix data structure and default Jacobian evaluation
     routine.  Whenever the nonlinear solver needs to compute the
     Jacobian matrix, it will call this routine.
      - Note that the final routine argument is the user-defined
        context that provides application-specific data for the
        Jacobian evaluation routine.
  */
  CHKERRQ(SNESSetJacobian(snes,J,J,FormJacobian,&ctx));
  CHKERRQ(SNESSetFromOptions(snes));
  CHKERRQ(SNESGetTolerances(snes,&abstol,&rtol,&stol,&maxit,&maxf));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"atol=%g, rtol=%g, stol=%g, maxit=%D, maxf=%D\n",(double)abstol,(double)rtol,(double)stol,maxit,maxf));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize application:
     Store forcing function of PDE and exact solution
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  {
    PetscScalarKokkosOffsetView FF,UU;
    CHKERRQ(DMDAVecGetKokkosOffsetViewWrite(ctx.da,F,&FF));
    CHKERRQ(DMDAVecGetKokkosOffsetViewWrite(ctx.da,U,&UU));
    Kokkos:: parallel_for (Kokkos::RangePolicy<>(FF.begin(0),FF.end(0)),KOKKOS_LAMBDA (int i) {
      PetscReal xp = i*ctx.h;
      FF(i) = 6.0*xp + pow(xp+1.e-12,6.0); /* +1.e-12 is to prevent 0^6 */
      UU(i) = xp*xp*xp;
    });
    CHKERRQ(DMDAVecRestoreKokkosOffsetViewWrite(ctx.da,F,&FF));
    CHKERRQ(DMDAVecRestoreKokkosOffsetViewWrite(ctx.da,U,&UU));
  }

  if (viewinitial) {
    CHKERRQ(VecView(U,NULL));
    CHKERRQ(VecView(F,NULL));
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
  CHKERRQ(FormInitialGuess(x));
  CHKERRQ(SNESSolve(snes,NULL,x));
  CHKERRQ(SNESGetIterationNumber(snes,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Number of SNES iterations = %D\n",its));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Check solution and clean up
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Check the error
  */
  CHKERRQ(VecAXPY(x,none,U));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g Iterations %D\n",(double)norm,its));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(VecDestroy(&U));
  CHKERRQ(VecDestroy(&F));
  CHKERRQ(MatDestroy(&J));
  CHKERRQ(SNESDestroy(&snes));
  CHKERRQ(DMDestroy(&ctx.da));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: kokkos_kernels

   test:
     requires: kokkos_kernels !complex !single
     nsize: 2
     args: -dm_vec_type kokkos -dm_mat_type aijkokkos -view_initial -snes_monitor
     output_file: output/ex3k_1.out

TEST*/
