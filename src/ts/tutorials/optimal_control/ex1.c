#include <petsctao.h>
#include <petscts.h>

typedef struct _n_aircraft *Aircraft;
struct _n_aircraft {
  TS        ts, quadts;
  Vec       V, W;   /* control variables V and W */
  PetscInt  nsteps; /* number of time steps */
  PetscReal ftime;
  Mat       A, H;
  Mat       Jacp, DRDU, DRDP;
  Vec       U, Lambda[1], Mup[1], Lambda2[1], Mup2[1], Dir;
  Vec       rhshp1[1], rhshp2[1], rhshp3[1], rhshp4[1], inthp1[1], inthp2[1], inthp3[1], inthp4[1];
  PetscReal lv, lw;
  PetscBool mf, eh;
};

PetscErrorCode FormObjFunctionGradient(Tao, Vec, PetscReal *, Vec, void *);
PetscErrorCode FormObjHessian(Tao, Vec, Mat, Mat, void *);
PetscErrorCode ComputeObjHessianWithSOA(Vec, PetscScalar[], Aircraft);
PetscErrorCode MatrixFreeObjHessian(Tao, Vec, Mat, Mat, void *);
PetscErrorCode MyMatMult(Mat, Vec, Vec);

static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  Aircraft           actx = (Aircraft)ctx;
  const PetscScalar *u, *v, *w;
  PetscScalar       *f;
  PetscInt           step;

  PetscFunctionBeginUser;
  PetscCall(TSGetStepNumber(ts, &step));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArrayRead(actx->V, &v));
  PetscCall(VecGetArrayRead(actx->W, &w));
  PetscCall(VecGetArray(F, &f));
  f[0] = v[step] * PetscCosReal(w[step]);
  f[1] = v[step] * PetscSinReal(w[step]);
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(actx->V, &v));
  PetscCall(VecRestoreArrayRead(actx->W, &w));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSJacobianP(TS ts, PetscReal t, Vec U, Mat A, void *ctx)
{
  Aircraft           actx = (Aircraft)ctx;
  const PetscScalar *u, *v, *w;
  PetscInt           step, rows[2] = {0, 1}, rowcol[2];
  PetscScalar        Jp[2][2];

  PetscFunctionBeginUser;
  PetscCall(MatZeroEntries(A));
  PetscCall(TSGetStepNumber(ts, &step));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArrayRead(actx->V, &v));
  PetscCall(VecGetArrayRead(actx->W, &w));

  Jp[0][0] = PetscCosReal(w[step]);
  Jp[0][1] = -v[step] * PetscSinReal(w[step]);
  Jp[1][0] = PetscSinReal(w[step]);
  Jp[1][1] = v[step] * PetscCosReal(w[step]);

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(actx->V, &v));
  PetscCall(VecRestoreArrayRead(actx->W, &w));

  rowcol[0] = 2 * step;
  rowcol[1] = 2 * step + 1;
  PetscCall(MatSetValues(A, 2, rows, 2, rowcol, &Jp[0][0], INSERT_VALUES));

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSHessianProductUU(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV, void *ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSHessianProductUP(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV, void *ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSHessianProductPU(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV, void *ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSHessianProductPP(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV, void *ctx)
{
  Aircraft           actx = (Aircraft)ctx;
  const PetscScalar *v, *w, *vl, *vr, *u;
  PetscScalar       *vhv;
  PetscScalar        dJpdP[2][2][2] = {{{0}}};
  PetscInt           step, i, j, k;

  PetscFunctionBeginUser;
  PetscCall(TSGetStepNumber(ts, &step));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArrayRead(actx->V, &v));
  PetscCall(VecGetArrayRead(actx->W, &w));
  PetscCall(VecGetArrayRead(Vl[0], &vl));
  PetscCall(VecGetArrayRead(Vr, &vr));
  PetscCall(VecSet(VHV[0], 0.0));
  PetscCall(VecGetArray(VHV[0], &vhv));

  dJpdP[0][0][1] = -PetscSinReal(w[step]);
  dJpdP[0][1][0] = -PetscSinReal(w[step]);
  dJpdP[0][1][1] = -v[step] * PetscCosReal(w[step]);
  dJpdP[1][0][1] = PetscCosReal(w[step]);
  dJpdP[1][1][0] = PetscCosReal(w[step]);
  dJpdP[1][1][1] = -v[step] * PetscSinReal(w[step]);

  for (j = 0; j < 2; j++) {
    vhv[2 * step + j] = 0;
    for (k = 0; k < 2; k++)
      for (i = 0; i < 2; i++) vhv[2 * step + j] += vl[i] * dJpdP[i][j][k] * vr[2 * step + k];
  }
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(Vl[0], &vl));
  PetscCall(VecRestoreArrayRead(Vr, &vr));
  PetscCall(VecRestoreArray(VHV[0], &vhv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Vl in NULL,updates to VHV must be added */
static PetscErrorCode IntegrandHessianProductUU(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV, void *ctx)
{
  Aircraft           actx = (Aircraft)ctx;
  const PetscScalar *v, *w, *vr, *u;
  PetscScalar       *vhv;
  PetscScalar        dRudU[2][2] = {{0}};
  PetscInt           step, j, k;

  PetscFunctionBeginUser;
  PetscCall(TSGetStepNumber(ts, &step));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArrayRead(actx->V, &v));
  PetscCall(VecGetArrayRead(actx->W, &w));
  PetscCall(VecGetArrayRead(Vr, &vr));
  PetscCall(VecGetArray(VHV[0], &vhv));

  dRudU[0][0] = 2.0;
  dRudU[1][1] = 2.0;

  for (j = 0; j < 2; j++) {
    vhv[j] = 0;
    for (k = 0; k < 2; k++) vhv[j] += dRudU[j][k] * vr[k];
  }
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(Vr, &vr));
  PetscCall(VecRestoreArray(VHV[0], &vhv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IntegrandHessianProductUP(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV, void *ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IntegrandHessianProductPU(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV, void *ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IntegrandHessianProductPP(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV, void *ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CostIntegrand(TS ts, PetscReal t, Vec U, Vec R, void *ctx)
{
  Aircraft           actx = (Aircraft)ctx;
  PetscScalar       *r;
  PetscReal          dx, dy;
  const PetscScalar *u;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(R, &r));
  dx   = u[0] - actx->lv * t * PetscCosReal(actx->lw);
  dy   = u[1] - actx->lv * t * PetscSinReal(actx->lw);
  r[0] = dx * dx + dy * dy;
  PetscCall(VecRestoreArray(R, &r));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DRDUJacobianTranspose(TS ts, PetscReal t, Vec U, Mat DRDU, Mat B, void *ctx)
{
  Aircraft           actx = (Aircraft)ctx;
  PetscScalar        drdu[2][1];
  const PetscScalar *u;
  PetscReal          dx, dy;
  PetscInt           row[] = {0, 1}, col[] = {0};

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(U, &u));
  dx         = u[0] - actx->lv * t * PetscCosReal(actx->lw);
  dy         = u[1] - actx->lv * t * PetscSinReal(actx->lw);
  drdu[0][0] = 2. * dx;
  drdu[1][0] = 2. * dy;
  PetscCall(MatSetValues(DRDU, 2, row, 1, col, &drdu[0][0], INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(MatAssemblyBegin(DRDU, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(DRDU, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DRDPJacobianTranspose(TS ts, PetscReal t, Vec U, Mat DRDP, void *ctx)
{
  PetscFunctionBegin;
  PetscCall(MatZeroEntries(DRDP));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Vec                P, PL, PU;
  struct _n_aircraft aircraft;
  PetscMPIInt        size;
  Tao                tao;
  KSP                ksp;
  PC                 pc;
  PetscScalar       *u, *p;
  PetscInt           i;

  /* Initialize program */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

  /* Parameter settings */
  aircraft.ftime  = 1.;            /* time interval in hour */
  aircraft.nsteps = 10;            /* number of steps */
  aircraft.lv     = 2.0;           /* leader speed in kmph */
  aircraft.lw     = PETSC_PI / 4.; /* leader heading angle */

  PetscCall(PetscOptionsGetReal(NULL, NULL, "-ftime", &aircraft.ftime, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nsteps", &aircraft.nsteps, NULL));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-matrixfree", &aircraft.mf));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-exacthessian", &aircraft.eh));

  /* Create TAO solver and set desired solution method */
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetType(tao, TAOBQNLS));

  /* Create necessary matrix and vectors, solve same ODE on every process */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &aircraft.A));
  PetscCall(MatSetSizes(aircraft.A, PETSC_DECIDE, PETSC_DECIDE, 2, 2));
  PetscCall(MatSetFromOptions(aircraft.A));
  PetscCall(MatSetUp(aircraft.A));
  /* this is to set explicit zeros along the diagonal of the matrix */
  PetscCall(MatAssemblyBegin(aircraft.A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(aircraft.A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatShift(aircraft.A, 1));
  PetscCall(MatShift(aircraft.A, -1));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &aircraft.Jacp));
  PetscCall(MatSetSizes(aircraft.Jacp, PETSC_DECIDE, PETSC_DECIDE, 2, 2 * aircraft.nsteps));
  PetscCall(MatSetFromOptions(aircraft.Jacp));
  PetscCall(MatSetUp(aircraft.Jacp));
  PetscCall(MatSetOption(aircraft.Jacp, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 2 * aircraft.nsteps, 1, NULL, &aircraft.DRDP));
  PetscCall(MatSetUp(aircraft.DRDP));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 2, 1, NULL, &aircraft.DRDU));
  PetscCall(MatSetUp(aircraft.DRDU));

  /* Create timestepping solver context */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &aircraft.ts));
  PetscCall(TSSetType(aircraft.ts, TSRK));
  PetscCall(TSSetRHSFunction(aircraft.ts, NULL, RHSFunction, &aircraft));
  PetscCall(TSSetRHSJacobian(aircraft.ts, aircraft.A, aircraft.A, TSComputeRHSJacobianConstant, &aircraft));
  PetscCall(TSSetRHSJacobianP(aircraft.ts, aircraft.Jacp, RHSJacobianP, &aircraft));
  PetscCall(TSSetExactFinalTime(aircraft.ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetEquationType(aircraft.ts, TS_EQ_ODE_EXPLICIT)); /* less Jacobian evaluations when adjoint BEuler is used, otherwise no effect */

  /* Set initial conditions */
  PetscCall(MatCreateVecs(aircraft.A, &aircraft.U, NULL));
  PetscCall(TSSetSolution(aircraft.ts, aircraft.U));
  PetscCall(VecGetArray(aircraft.U, &u));
  u[0] = 1.5;
  u[1] = 0;
  PetscCall(VecRestoreArray(aircraft.U, &u));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &aircraft.V));
  PetscCall(VecSetSizes(aircraft.V, PETSC_DECIDE, aircraft.nsteps));
  PetscCall(VecSetUp(aircraft.V));
  PetscCall(VecDuplicate(aircraft.V, &aircraft.W));
  PetscCall(VecSet(aircraft.V, 1.));
  PetscCall(VecSet(aircraft.W, PETSC_PI / 4.));

  /* Save trajectory of solution so that TSAdjointSolve() may be used */
  PetscCall(TSSetSaveTrajectory(aircraft.ts));

  /* Set sensitivity context */
  PetscCall(TSCreateQuadratureTS(aircraft.ts, PETSC_FALSE, &aircraft.quadts));
  PetscCall(TSSetRHSFunction(aircraft.quadts, NULL, (TSRHSFunction)CostIntegrand, &aircraft));
  PetscCall(TSSetRHSJacobian(aircraft.quadts, aircraft.DRDU, aircraft.DRDU, (TSRHSJacobian)DRDUJacobianTranspose, &aircraft));
  PetscCall(TSSetRHSJacobianP(aircraft.quadts, aircraft.DRDP, (TSRHSJacobianP)DRDPJacobianTranspose, &aircraft));
  PetscCall(MatCreateVecs(aircraft.A, &aircraft.Lambda[0], NULL));
  PetscCall(MatCreateVecs(aircraft.Jacp, &aircraft.Mup[0], NULL));
  if (aircraft.eh) {
    PetscCall(MatCreateVecs(aircraft.A, &aircraft.rhshp1[0], NULL));
    PetscCall(MatCreateVecs(aircraft.A, &aircraft.rhshp2[0], NULL));
    PetscCall(MatCreateVecs(aircraft.Jacp, &aircraft.rhshp3[0], NULL));
    PetscCall(MatCreateVecs(aircraft.Jacp, &aircraft.rhshp4[0], NULL));
    PetscCall(MatCreateVecs(aircraft.DRDU, &aircraft.inthp1[0], NULL));
    PetscCall(MatCreateVecs(aircraft.DRDU, &aircraft.inthp2[0], NULL));
    PetscCall(MatCreateVecs(aircraft.DRDP, &aircraft.inthp3[0], NULL));
    PetscCall(MatCreateVecs(aircraft.DRDP, &aircraft.inthp4[0], NULL));
    PetscCall(MatCreateVecs(aircraft.Jacp, &aircraft.Dir, NULL));
    PetscCall(TSSetRHSHessianProduct(aircraft.ts, aircraft.rhshp1, RHSHessianProductUU, aircraft.rhshp2, RHSHessianProductUP, aircraft.rhshp3, RHSHessianProductPU, aircraft.rhshp4, RHSHessianProductPP, &aircraft));
    PetscCall(TSSetRHSHessianProduct(aircraft.quadts, aircraft.inthp1, IntegrandHessianProductUU, aircraft.inthp2, IntegrandHessianProductUP, aircraft.inthp3, IntegrandHessianProductPU, aircraft.inthp4, IntegrandHessianProductPP, &aircraft));
    PetscCall(MatCreateVecs(aircraft.A, &aircraft.Lambda2[0], NULL));
    PetscCall(MatCreateVecs(aircraft.Jacp, &aircraft.Mup2[0], NULL));
  }
  PetscCall(TSSetFromOptions(aircraft.ts));
  PetscCall(TSSetMaxTime(aircraft.ts, aircraft.ftime));
  PetscCall(TSSetTimeStep(aircraft.ts, aircraft.ftime / aircraft.nsteps));

  /* Set initial solution guess */
  PetscCall(MatCreateVecs(aircraft.Jacp, &P, NULL));
  PetscCall(VecGetArray(P, &p));
  for (i = 0; i < aircraft.nsteps; i++) {
    p[2 * i]     = 2.0;
    p[2 * i + 1] = PETSC_PI / 2.0;
  }
  PetscCall(VecRestoreArray(P, &p));
  PetscCall(VecDuplicate(P, &PU));
  PetscCall(VecDuplicate(P, &PL));
  PetscCall(VecGetArray(PU, &p));
  for (i = 0; i < aircraft.nsteps; i++) {
    p[2 * i]     = 2.0;
    p[2 * i + 1] = PETSC_PI;
  }
  PetscCall(VecRestoreArray(PU, &p));
  PetscCall(VecGetArray(PL, &p));
  for (i = 0; i < aircraft.nsteps; i++) {
    p[2 * i]     = 0.0;
    p[2 * i + 1] = -PETSC_PI;
  }
  PetscCall(VecRestoreArray(PL, &p));

  PetscCall(TaoSetSolution(tao, P));
  PetscCall(TaoSetVariableBounds(tao, PL, PU));
  /* Set routine for function and gradient evaluation */
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormObjFunctionGradient, (void *)&aircraft));

  if (aircraft.eh) {
    if (aircraft.mf) {
      PetscCall(MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 2 * aircraft.nsteps, 2 * aircraft.nsteps, (void *)&aircraft, &aircraft.H));
      PetscCall(MatShellSetOperation(aircraft.H, MATOP_MULT, (void (*)(void))MyMatMult));
      PetscCall(MatSetOption(aircraft.H, MAT_SYMMETRIC, PETSC_TRUE));
      PetscCall(TaoSetHessian(tao, aircraft.H, aircraft.H, MatrixFreeObjHessian, (void *)&aircraft));
    } else {
      PetscCall(MatCreateDense(MPI_COMM_WORLD, PETSC_DETERMINE, PETSC_DETERMINE, 2 * aircraft.nsteps, 2 * aircraft.nsteps, NULL, &(aircraft.H)));
      PetscCall(MatSetOption(aircraft.H, MAT_SYMMETRIC, PETSC_TRUE));
      PetscCall(TaoSetHessian(tao, aircraft.H, aircraft.H, FormObjHessian, (void *)&aircraft));
    }
  }

  /* Check for any TAO command line options */
  PetscCall(TaoGetKSP(tao, &ksp));
  if (ksp) {
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCNONE));
  }
  PetscCall(TaoSetFromOptions(tao));

  PetscCall(TaoSolve(tao));
  PetscCall(VecView(P, PETSC_VIEWER_STDOUT_WORLD));

  /* Free TAO data structures */
  PetscCall(TaoDestroy(&tao));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSDestroy(&aircraft.ts));
  PetscCall(MatDestroy(&aircraft.A));
  PetscCall(VecDestroy(&aircraft.U));
  PetscCall(VecDestroy(&aircraft.V));
  PetscCall(VecDestroy(&aircraft.W));
  PetscCall(VecDestroy(&P));
  PetscCall(VecDestroy(&PU));
  PetscCall(VecDestroy(&PL));
  PetscCall(MatDestroy(&aircraft.Jacp));
  PetscCall(MatDestroy(&aircraft.DRDU));
  PetscCall(MatDestroy(&aircraft.DRDP));
  PetscCall(VecDestroy(&aircraft.Lambda[0]));
  PetscCall(VecDestroy(&aircraft.Mup[0]));
  PetscCall(VecDestroy(&P));
  if (aircraft.eh) {
    PetscCall(VecDestroy(&aircraft.Lambda2[0]));
    PetscCall(VecDestroy(&aircraft.Mup2[0]));
    PetscCall(VecDestroy(&aircraft.Dir));
    PetscCall(VecDestroy(&aircraft.rhshp1[0]));
    PetscCall(VecDestroy(&aircraft.rhshp2[0]));
    PetscCall(VecDestroy(&aircraft.rhshp3[0]));
    PetscCall(VecDestroy(&aircraft.rhshp4[0]));
    PetscCall(VecDestroy(&aircraft.inthp1[0]));
    PetscCall(VecDestroy(&aircraft.inthp2[0]));
    PetscCall(VecDestroy(&aircraft.inthp3[0]));
    PetscCall(VecDestroy(&aircraft.inthp4[0]));
    PetscCall(MatDestroy(&aircraft.H));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*
   FormObjFunctionGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   P   - the input vector
   ctx - optional aircraft-defined context, as set by TaoSetObjectiveAndGradient()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient
*/
PetscErrorCode FormObjFunctionGradient(Tao tao, Vec P, PetscReal *f, Vec G, void *ctx)
{
  Aircraft           actx = (Aircraft)ctx;
  TS                 ts   = actx->ts;
  Vec                Q;
  const PetscScalar *p, *q;
  PetscScalar       *u, *v, *w;
  PetscInt           i;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(P, &p));
  PetscCall(VecGetArray(actx->V, &v));
  PetscCall(VecGetArray(actx->W, &w));
  for (i = 0; i < actx->nsteps; i++) {
    v[i] = p[2 * i];
    w[i] = p[2 * i + 1];
  }
  PetscCall(VecRestoreArrayRead(P, &p));
  PetscCall(VecRestoreArray(actx->V, &v));
  PetscCall(VecRestoreArray(actx->W, &w));

  PetscCall(TSSetTime(ts, 0.0));
  PetscCall(TSSetStepNumber(ts, 0));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetTimeStep(ts, actx->ftime / actx->nsteps));

  /* reinitialize system state */
  PetscCall(VecGetArray(actx->U, &u));
  u[0] = 2.0;
  u[1] = 0;
  PetscCall(VecRestoreArray(actx->U, &u));

  /* reinitialize the integral value */
  PetscCall(TSGetCostIntegral(ts, &Q));
  PetscCall(VecSet(Q, 0.0));

  PetscCall(TSSolve(ts, actx->U));

  /* Reset initial conditions for the adjoint integration */
  PetscCall(VecSet(actx->Lambda[0], 0.0));
  PetscCall(VecSet(actx->Mup[0], 0.0));
  PetscCall(TSSetCostGradients(ts, 1, actx->Lambda, actx->Mup));

  PetscCall(TSAdjointSolve(ts));
  PetscCall(VecCopy(actx->Mup[0], G));
  PetscCall(TSGetCostIntegral(ts, &Q));
  PetscCall(VecGetArrayRead(Q, &q));
  *f = q[0];
  PetscCall(VecRestoreArrayRead(Q, &q));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormObjHessian(Tao tao, Vec P, Mat H, Mat Hpre, void *ctx)
{
  Aircraft           actx = (Aircraft)ctx;
  const PetscScalar *p;
  PetscScalar       *harr, *v, *w, one = 1.0;
  PetscInt           ind[1];
  PetscInt          *cols, i;
  Vec                Dir;

  PetscFunctionBeginUser;
  /* set up control parameters */
  PetscCall(VecGetArrayRead(P, &p));
  PetscCall(VecGetArray(actx->V, &v));
  PetscCall(VecGetArray(actx->W, &w));
  for (i = 0; i < actx->nsteps; i++) {
    v[i] = p[2 * i];
    w[i] = p[2 * i + 1];
  }
  PetscCall(VecRestoreArrayRead(P, &p));
  PetscCall(VecRestoreArray(actx->V, &v));
  PetscCall(VecRestoreArray(actx->W, &w));

  PetscCall(PetscMalloc1(2 * actx->nsteps, &harr));
  PetscCall(PetscMalloc1(2 * actx->nsteps, &cols));
  for (i = 0; i < 2 * actx->nsteps; i++) cols[i] = i;
  PetscCall(VecDuplicate(P, &Dir));
  for (i = 0; i < 2 * actx->nsteps; i++) {
    ind[0] = i;
    PetscCall(VecSet(Dir, 0.0));
    PetscCall(VecSetValues(Dir, 1, ind, &one, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(Dir));
    PetscCall(VecAssemblyEnd(Dir));
    PetscCall(ComputeObjHessianWithSOA(Dir, harr, actx));
    PetscCall(MatSetValues(H, 1, ind, 2 * actx->nsteps, cols, harr, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
    if (H != Hpre) {
      PetscCall(MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY));
    }
  }
  PetscCall(PetscFree(cols));
  PetscCall(PetscFree(harr));
  PetscCall(VecDestroy(&Dir));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatrixFreeObjHessian(Tao tao, Vec P, Mat H, Mat Hpre, void *ctx)
{
  Aircraft           actx = (Aircraft)ctx;
  PetscScalar       *v, *w;
  const PetscScalar *p;
  PetscInt           i;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(P, &p));
  PetscCall(VecGetArray(actx->V, &v));
  PetscCall(VecGetArray(actx->W, &w));
  for (i = 0; i < actx->nsteps; i++) {
    v[i] = p[2 * i];
    w[i] = p[2 * i + 1];
  }
  PetscCall(VecRestoreArrayRead(P, &p));
  PetscCall(VecRestoreArray(actx->V, &v));
  PetscCall(VecRestoreArray(actx->W, &w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MyMatMult(Mat H_shell, Vec X, Vec Y)
{
  PetscScalar *y;
  void        *ptr;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(H_shell, &ptr));
  PetscCall(VecGetArray(Y, &y));
  PetscCall(ComputeObjHessianWithSOA(X, y, (Aircraft)ptr));
  PetscCall(VecRestoreArray(Y, &y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeObjHessianWithSOA(Vec Dir, PetscScalar arr[], Aircraft actx)
{
  TS                 ts = actx->ts;
  const PetscScalar *z_ptr;
  PetscScalar       *u;
  Vec                Q;
  PetscInt           i;

  PetscFunctionBeginUser;
  /* Reset TSAdjoint so that AdjointSetUp will be called again */
  PetscCall(TSAdjointReset(ts));

  PetscCall(TSSetTime(ts, 0.0));
  PetscCall(TSSetStepNumber(ts, 0));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetTimeStep(ts, actx->ftime / actx->nsteps));
  PetscCall(TSSetCostHessianProducts(actx->ts, 1, actx->Lambda2, actx->Mup2, Dir));

  /* reinitialize system state */
  PetscCall(VecGetArray(actx->U, &u));
  u[0] = 2.0;
  u[1] = 0;
  PetscCall(VecRestoreArray(actx->U, &u));

  /* reinitialize the integral value */
  PetscCall(TSGetCostIntegral(ts, &Q));
  PetscCall(VecSet(Q, 0.0));

  /* initialize tlm variable */
  PetscCall(MatZeroEntries(actx->Jacp));
  PetscCall(TSAdjointSetForward(ts, actx->Jacp));

  PetscCall(TSSolve(ts, actx->U));

  /* Set terminal conditions for first- and second-order adjonts */
  PetscCall(VecSet(actx->Lambda[0], 0.0));
  PetscCall(VecSet(actx->Mup[0], 0.0));
  PetscCall(VecSet(actx->Lambda2[0], 0.0));
  PetscCall(VecSet(actx->Mup2[0], 0.0));
  PetscCall(TSSetCostGradients(ts, 1, actx->Lambda, actx->Mup));

  PetscCall(TSGetCostIntegral(ts, &Q));

  /* Reset initial conditions for the adjoint integration */
  PetscCall(TSAdjointSolve(ts));

  /* initial condition does not depend on p, so that lambda is not needed to assemble G */
  PetscCall(VecGetArrayRead(actx->Mup2[0], &z_ptr));
  for (i = 0; i < 2 * actx->nsteps; i++) arr[i] = z_ptr[i];
  PetscCall(VecRestoreArrayRead(actx->Mup2[0], &z_ptr));

  /* Disable second-order adjoint mode */
  PetscCall(TSAdjointReset(ts));
  PetscCall(TSAdjointResetForward(ts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST
    build:
      requires: !complex !single

    test:
      args:  -ts_adapt_type none -ts_type rk -ts_rk_type 3 -viewer_binary_skip_info -tao_monitor -tao_gatol 1e-7

    test:
      suffix: 2
      args:  -ts_adapt_type none -ts_type rk -ts_rk_type 3 -viewer_binary_skip_info -tao_monitor -tao_view -tao_type bntr -tao_bnk_pc_type none -exacthessian

    test:
      suffix: 3
      args:  -ts_adapt_type none -ts_type rk -ts_rk_type 3 -viewer_binary_skip_info -tao_monitor -tao_view -tao_type bntr -tao_bnk_pc_type none -exacthessian -matrixfree
TEST*/
