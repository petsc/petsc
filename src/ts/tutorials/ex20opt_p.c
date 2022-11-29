
static char help[] = "Solves the van der Pol equation.\n\
Input parameters include:\n";

/* ------------------------------------------------------------------------

  Notes:
  This code demonstrates how to solve a DAE-constrained optimization problem with TAO, TSAdjoint and TS.
  The nonlinear problem is written in a DAE equivalent form.
  The objective is to minimize the difference between observation and model prediction by finding an optimal value for parameter \mu.
  The gradient is computed with the discrete adjoint of an implicit theta method, see ex20adj.c for details.
  ------------------------------------------------------------------------- */
#include <petsctao.h>
#include <petscts.h>

typedef struct _n_User *User;
struct _n_User {
  TS        ts;
  PetscReal mu;
  PetscReal next_output;

  /* Sensitivity analysis support */
  PetscReal ftime;
  Mat       A;                    /* Jacobian matrix */
  Mat       Jacp;                 /* JacobianP matrix */
  Mat       H;                    /* Hessian matrix for optimization */
  Vec       U, Lambda[1], Mup[1]; /* adjoint variables */
  Vec       Lambda2[1], Mup2[1];  /* second-order adjoint variables */
  Vec       Ihp1[1];              /* working space for Hessian evaluations */
  Vec       Ihp2[1];              /* working space for Hessian evaluations */
  Vec       Ihp3[1];              /* working space for Hessian evaluations */
  Vec       Ihp4[1];              /* working space for Hessian evaluations */
  Vec       Dir;                  /* direction vector */
  PetscReal ob[2];                /* observation used by the cost function */
  PetscBool implicitform;         /* implicit ODE? */
};

PetscErrorCode FormFunctionGradient(Tao, Vec, PetscReal *, Vec, void *);
PetscErrorCode FormHessian(Tao, Vec, Mat, Mat, void *);
PetscErrorCode Adjoint2(Vec, PetscScalar[], User);

/* ----------------------- Explicit form of the ODE  -------------------- */

static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  User               user = (User)ctx;
  PetscScalar       *f;
  const PetscScalar *u;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(F, &f));
  f[0] = u[1];
  f[1] = user->mu * ((1. - u[0] * u[0]) * u[1] - u[0]);
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec U, Mat A, Mat B, void *ctx)
{
  User               user     = (User)ctx;
  PetscReal          mu       = user->mu;
  PetscInt           rowcol[] = {0, 1};
  PetscScalar        J[2][2];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));

  J[0][0] = 0;
  J[1][0] = -mu * (2.0 * u[1] * u[0] + 1.);
  J[0][1] = 1.0;
  J[1][1] = mu * (1.0 - u[0] * u[0]);
  PetscCall(MatSetValues(A, 2, rowcol, 2, rowcol, &J[0][0], INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  if (B && A != B) {
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  }

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSHessianProductUU(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV, void *ctx)
{
  const PetscScalar *vl, *vr, *u;
  PetscScalar       *vhv;
  PetscScalar        dJdU[2][2][2] = {{{0}}};
  PetscInt           i, j, k;
  User               user = (User)ctx;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArrayRead(Vl[0], &vl));
  PetscCall(VecGetArrayRead(Vr, &vr));
  PetscCall(VecGetArray(VHV[0], &vhv));

  dJdU[1][0][0] = -2. * user->mu * u[1];
  dJdU[1][1][0] = -2. * user->mu * u[0];
  dJdU[1][0][1] = -2. * user->mu * u[0];
  for (j = 0; j < 2; j++) {
    vhv[j] = 0;
    for (k = 0; k < 2; k++)
      for (i = 0; i < 2; i++) vhv[j] += vl[i] * dJdU[i][j][k] * vr[k];
  }

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(Vl[0], &vl));
  PetscCall(VecRestoreArrayRead(Vr, &vr));
  PetscCall(VecRestoreArray(VHV[0], &vhv));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSHessianProductUP(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV, void *ctx)
{
  const PetscScalar *vl, *vr, *u;
  PetscScalar       *vhv;
  PetscScalar        dJdP[2][2][1] = {{{0}}};
  PetscInt           i, j, k;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArrayRead(Vl[0], &vl));
  PetscCall(VecGetArrayRead(Vr, &vr));
  PetscCall(VecGetArray(VHV[0], &vhv));

  dJdP[1][0][0] = -(1. + 2. * u[0] * u[1]);
  dJdP[1][1][0] = 1. - u[0] * u[0];
  for (j = 0; j < 2; j++) {
    vhv[j] = 0;
    for (k = 0; k < 1; k++)
      for (i = 0; i < 2; i++) vhv[j] += vl[i] * dJdP[i][j][k] * vr[k];
  }

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(Vl[0], &vl));
  PetscCall(VecRestoreArrayRead(Vr, &vr));
  PetscCall(VecRestoreArray(VHV[0], &vhv));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSHessianProductPU(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV, void *ctx)
{
  const PetscScalar *vl, *vr, *u;
  PetscScalar       *vhv;
  PetscScalar        dJdU[2][1][2] = {{{0}}};
  PetscInt           i, j, k;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArrayRead(Vl[0], &vl));
  PetscCall(VecGetArrayRead(Vr, &vr));
  PetscCall(VecGetArray(VHV[0], &vhv));

  dJdU[1][0][0] = -1. - 2. * u[1] * u[0];
  dJdU[1][0][1] = 1. - u[0] * u[0];
  for (j = 0; j < 1; j++) {
    vhv[j] = 0;
    for (k = 0; k < 2; k++)
      for (i = 0; i < 2; i++) vhv[j] += vl[i] * dJdU[i][j][k] * vr[k];
  }

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(Vl[0], &vl));
  PetscCall(VecRestoreArrayRead(Vr, &vr));
  PetscCall(VecRestoreArray(VHV[0], &vhv));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSHessianProductPP(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV, void *ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(0);
}

/* ----------------------- Implicit form of the ODE  -------------------- */

static PetscErrorCode IFunction(TS ts, PetscReal t, Vec U, Vec Udot, Vec F, void *ctx)
{
  User               user = (User)ctx;
  PetscScalar       *f;
  const PetscScalar *u, *udot;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArrayRead(Udot, &udot));
  PetscCall(VecGetArray(F, &f));

  f[0] = udot[0] - u[1];
  f[1] = udot[1] - user->mu * ((1.0 - u[0] * u[0]) * u[1] - u[0]);

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(Udot, &udot));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobian(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal a, Mat A, Mat B, void *ctx)
{
  User               user     = (User)ctx;
  PetscInt           rowcol[] = {0, 1};
  PetscScalar        J[2][2];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));

  J[0][0] = a;
  J[0][1] = -1.0;
  J[1][0] = user->mu * (1.0 + 2.0 * u[0] * u[1]);
  J[1][1] = a - user->mu * (1.0 - u[0] * u[0]);
  PetscCall(MatSetValues(B, 2, rowcol, 2, rowcol, &J[0][0], INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  }

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobianP(TS ts, PetscReal t, Vec U, Mat A, void *ctx)
{
  PetscInt           row[] = {0, 1}, col[] = {0};
  PetscScalar        J[2][1];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));

  J[0][0] = 0;
  J[1][0] = (1. - u[0] * u[0]) * u[1] - u[0];
  PetscCall(MatSetValues(A, 2, row, 1, col, &J[0][0], INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscFunctionReturn(0);
}

static PetscErrorCode IHessianProductUU(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV, void *ctx)
{
  const PetscScalar *vl, *vr, *u;
  PetscScalar       *vhv;
  PetscScalar        dJdU[2][2][2] = {{{0}}};
  PetscInt           i, j, k;
  User               user = (User)ctx;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArrayRead(Vl[0], &vl));
  PetscCall(VecGetArrayRead(Vr, &vr));
  PetscCall(VecGetArray(VHV[0], &vhv));

  dJdU[1][0][0] = 2. * user->mu * u[1];
  dJdU[1][1][0] = 2. * user->mu * u[0];
  dJdU[1][0][1] = 2. * user->mu * u[0];
  for (j = 0; j < 2; j++) {
    vhv[j] = 0;
    for (k = 0; k < 2; k++)
      for (i = 0; i < 2; i++) vhv[j] += vl[i] * dJdU[i][j][k] * vr[k];
  }

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(Vl[0], &vl));
  PetscCall(VecRestoreArrayRead(Vr, &vr));
  PetscCall(VecRestoreArray(VHV[0], &vhv));
  PetscFunctionReturn(0);
}

static PetscErrorCode IHessianProductUP(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV, void *ctx)
{
  const PetscScalar *vl, *vr, *u;
  PetscScalar       *vhv;
  PetscScalar        dJdP[2][2][1] = {{{0}}};
  PetscInt           i, j, k;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArrayRead(Vl[0], &vl));
  PetscCall(VecGetArrayRead(Vr, &vr));
  PetscCall(VecGetArray(VHV[0], &vhv));

  dJdP[1][0][0] = 1. + 2. * u[0] * u[1];
  dJdP[1][1][0] = u[0] * u[0] - 1.;
  for (j = 0; j < 2; j++) {
    vhv[j] = 0;
    for (k = 0; k < 1; k++)
      for (i = 0; i < 2; i++) vhv[j] += vl[i] * dJdP[i][j][k] * vr[k];
  }

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(Vl[0], &vl));
  PetscCall(VecRestoreArrayRead(Vr, &vr));
  PetscCall(VecRestoreArray(VHV[0], &vhv));
  PetscFunctionReturn(0);
}

static PetscErrorCode IHessianProductPU(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV, void *ctx)
{
  const PetscScalar *vl, *vr, *u;
  PetscScalar       *vhv;
  PetscScalar        dJdU[2][1][2] = {{{0}}};
  PetscInt           i, j, k;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArrayRead(Vl[0], &vl));
  PetscCall(VecGetArrayRead(Vr, &vr));
  PetscCall(VecGetArray(VHV[0], &vhv));

  dJdU[1][0][0] = 1. + 2. * u[1] * u[0];
  dJdU[1][0][1] = u[0] * u[0] - 1.;
  for (j = 0; j < 1; j++) {
    vhv[j] = 0;
    for (k = 0; k < 2; k++)
      for (i = 0; i < 2; i++) vhv[j] += vl[i] * dJdU[i][j][k] * vr[k];
  }

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(Vl[0], &vl));
  PetscCall(VecRestoreArrayRead(Vr, &vr));
  PetscCall(VecRestoreArray(VHV[0], &vhv));
  PetscFunctionReturn(0);
}

static PetscErrorCode IHessianProductPP(TS ts, PetscReal t, Vec U, Vec *Vl, Vec Vr, Vec *VHV, void *ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(0);
}

/* Monitor timesteps and use interpolation to output at integer multiples of 0.1 */
static PetscErrorCode Monitor(TS ts, PetscInt step, PetscReal t, Vec X, void *ctx)
{
  const PetscScalar *x;
  PetscReal          tfinal, dt;
  User               user = (User)ctx;
  Vec                interpolatedX;

  PetscFunctionBeginUser;
  PetscCall(TSGetTimeStep(ts, &dt));
  PetscCall(TSGetMaxTime(ts, &tfinal));

  while (user->next_output <= t && user->next_output <= tfinal) {
    PetscCall(VecDuplicate(X, &interpolatedX));
    PetscCall(TSInterpolate(ts, user->next_output, interpolatedX));
    PetscCall(VecGetArrayRead(interpolatedX, &x));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[%g] %" PetscInt_FMT " TS %g (dt = %g) X %g %g\n", (double)user->next_output, step, (double)t, (double)dt, (double)PetscRealPart(x[0]), (double)PetscRealPart(x[1])));
    PetscCall(VecRestoreArrayRead(interpolatedX, &x));
    PetscCall(VecDestroy(&interpolatedX));
    user->next_output += PetscRealConstant(0.1);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  Vec                P;
  PetscBool          monitor = PETSC_FALSE;
  PetscScalar       *x_ptr;
  const PetscScalar *y_ptr;
  PetscMPIInt        size;
  struct _n_User     user;
  Tao                tao;
  KSP                ksp;
  PC                 pc;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

  /* Create TAO solver and set desired solution method */
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetType(tao, TAOBQNLS));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.next_output  = 0.0;
  user.mu           = PetscRealConstant(1.0e3);
  user.ftime        = PetscRealConstant(0.5);
  user.implicitform = PETSC_TRUE;

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-monitor", &monitor, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-mu", &user.mu, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-implicitform", &user.implicitform, NULL));

  /* Create necessary matrix and vectors, solve same ODE on every process */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &user.A));
  PetscCall(MatSetSizes(user.A, PETSC_DECIDE, PETSC_DECIDE, 2, 2));
  PetscCall(MatSetFromOptions(user.A));
  PetscCall(MatSetUp(user.A));
  PetscCall(MatCreateVecs(user.A, &user.U, NULL));
  PetscCall(MatCreateVecs(user.A, &user.Lambda[0], NULL));
  PetscCall(MatCreateVecs(user.A, &user.Lambda2[0], NULL));
  PetscCall(MatCreateVecs(user.A, &user.Ihp1[0], NULL));
  PetscCall(MatCreateVecs(user.A, &user.Ihp2[0], NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &user.Jacp));
  PetscCall(MatSetSizes(user.Jacp, PETSC_DECIDE, PETSC_DECIDE, 2, 1));
  PetscCall(MatSetFromOptions(user.Jacp));
  PetscCall(MatSetUp(user.Jacp));
  PetscCall(MatCreateVecs(user.Jacp, &user.Dir, NULL));
  PetscCall(MatCreateVecs(user.Jacp, &user.Mup[0], NULL));
  PetscCall(MatCreateVecs(user.Jacp, &user.Mup2[0], NULL));
  PetscCall(MatCreateVecs(user.Jacp, &user.Ihp3[0], NULL));
  PetscCall(MatCreateVecs(user.Jacp, &user.Ihp4[0], NULL));

  /* Create timestepping solver context */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &user.ts));
  PetscCall(TSSetEquationType(user.ts, TS_EQ_ODE_EXPLICIT)); /* less Jacobian evaluations when adjoint BEuler is used, otherwise no effect */
  if (user.implicitform) {
    PetscCall(TSSetIFunction(user.ts, NULL, IFunction, &user));
    PetscCall(TSSetIJacobian(user.ts, user.A, user.A, IJacobian, &user));
    PetscCall(TSSetType(user.ts, TSCN));
  } else {
    PetscCall(TSSetRHSFunction(user.ts, NULL, RHSFunction, &user));
    PetscCall(TSSetRHSJacobian(user.ts, user.A, user.A, RHSJacobian, &user));
    PetscCall(TSSetType(user.ts, TSRK));
  }
  PetscCall(TSSetRHSJacobianP(user.ts, user.Jacp, RHSJacobianP, &user));
  PetscCall(TSSetMaxTime(user.ts, user.ftime));
  PetscCall(TSSetExactFinalTime(user.ts, TS_EXACTFINALTIME_MATCHSTEP));

  if (monitor) PetscCall(TSMonitorSet(user.ts, Monitor, &user, NULL));

  /* Set ODE initial conditions */
  PetscCall(VecGetArray(user.U, &x_ptr));
  x_ptr[0] = 2.0;
  x_ptr[1] = -2.0 / 3.0 + 10.0 / (81.0 * user.mu) - 292.0 / (2187.0 * user.mu * user.mu);
  PetscCall(VecRestoreArray(user.U, &x_ptr));
  PetscCall(TSSetTimeStep(user.ts, PetscRealConstant(0.001)));

  /* Set runtime options */
  PetscCall(TSSetFromOptions(user.ts));

  PetscCall(TSSolve(user.ts, user.U));
  PetscCall(VecGetArrayRead(user.U, &y_ptr));
  user.ob[0] = y_ptr[0];
  user.ob[1] = y_ptr[1];
  PetscCall(VecRestoreArrayRead(user.U, &y_ptr));

  /* Save trajectory of solution so that TSAdjointSolve() may be used.
     Skip checkpointing for the first TSSolve since no adjoint run follows it.
   */
  PetscCall(TSSetSaveTrajectory(user.ts));

  /* Optimization starts */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &user.H));
  PetscCall(MatSetSizes(user.H, PETSC_DECIDE, PETSC_DECIDE, 1, 1));
  PetscCall(MatSetUp(user.H)); /* Hessian should be symmetric. Do we need to do MatSetOption(user.H,MAT_SYMMETRIC,PETSC_TRUE) ? */

  /* Set initial solution guess */
  PetscCall(MatCreateVecs(user.Jacp, &P, NULL));
  PetscCall(VecGetArray(P, &x_ptr));
  x_ptr[0] = PetscRealConstant(1.2);
  PetscCall(VecRestoreArray(P, &x_ptr));
  PetscCall(TaoSetSolution(tao, P));

  /* Set routine for function and gradient evaluation */
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormFunctionGradient, (void *)&user));
  PetscCall(TaoSetHessian(tao, user.H, user.H, FormHessian, (void *)&user));

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
  PetscCall(MatDestroy(&user.H));
  PetscCall(MatDestroy(&user.A));
  PetscCall(VecDestroy(&user.U));
  PetscCall(MatDestroy(&user.Jacp));
  PetscCall(VecDestroy(&user.Lambda[0]));
  PetscCall(VecDestroy(&user.Mup[0]));
  PetscCall(VecDestroy(&user.Lambda2[0]));
  PetscCall(VecDestroy(&user.Mup2[0]));
  PetscCall(VecDestroy(&user.Ihp1[0]));
  PetscCall(VecDestroy(&user.Ihp2[0]));
  PetscCall(VecDestroy(&user.Ihp3[0]));
  PetscCall(VecDestroy(&user.Ihp4[0]));
  PetscCall(VecDestroy(&user.Dir));
  PetscCall(TSDestroy(&user.ts));
  PetscCall(VecDestroy(&P));
  PetscCall(PetscFinalize());
  return 0;
}

/* ------------------------------------------------------------------ */
/*
   FormFunctionGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   X   - the input vector
   ptr - optional user-defined context, as set by TaoSetObjectiveAndGradient()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient
*/
PetscErrorCode FormFunctionGradient(Tao tao, Vec P, PetscReal *f, Vec G, void *ctx)
{
  User               user_ptr = (User)ctx;
  TS                 ts       = user_ptr->ts;
  PetscScalar       *x_ptr, *g;
  const PetscScalar *y_ptr;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(P, &y_ptr));
  user_ptr->mu = y_ptr[0];
  PetscCall(VecRestoreArrayRead(P, &y_ptr));

  PetscCall(TSSetTime(ts, 0.0));
  PetscCall(TSSetStepNumber(ts, 0));
  PetscCall(TSSetTimeStep(ts, PetscRealConstant(0.001))); /* can be overwritten by command line options */
  PetscCall(TSSetFromOptions(ts));
  PetscCall(VecGetArray(user_ptr->U, &x_ptr));
  x_ptr[0] = 2.0;
  x_ptr[1] = -2.0 / 3.0 + 10.0 / (81.0 * user_ptr->mu) - 292.0 / (2187.0 * user_ptr->mu * user_ptr->mu);
  PetscCall(VecRestoreArray(user_ptr->U, &x_ptr));

  PetscCall(TSSolve(ts, user_ptr->U));

  PetscCall(VecGetArrayRead(user_ptr->U, &y_ptr));
  *f = (y_ptr[0] - user_ptr->ob[0]) * (y_ptr[0] - user_ptr->ob[0]) + (y_ptr[1] - user_ptr->ob[1]) * (y_ptr[1] - user_ptr->ob[1]);

  /*   Reset initial conditions for the adjoint integration */
  PetscCall(VecGetArray(user_ptr->Lambda[0], &x_ptr));
  x_ptr[0] = 2. * (y_ptr[0] - user_ptr->ob[0]);
  x_ptr[1] = 2. * (y_ptr[1] - user_ptr->ob[1]);
  PetscCall(VecRestoreArrayRead(user_ptr->U, &y_ptr));
  PetscCall(VecRestoreArray(user_ptr->Lambda[0], &x_ptr));

  PetscCall(VecGetArray(user_ptr->Mup[0], &x_ptr));
  x_ptr[0] = 0.0;
  PetscCall(VecRestoreArray(user_ptr->Mup[0], &x_ptr));
  PetscCall(TSSetCostGradients(ts, 1, user_ptr->Lambda, user_ptr->Mup));

  PetscCall(TSAdjointSolve(ts));

  PetscCall(VecGetArray(user_ptr->Mup[0], &x_ptr));
  PetscCall(VecGetArrayRead(user_ptr->Lambda[0], &y_ptr));
  PetscCall(VecGetArray(G, &g));
  g[0] = y_ptr[1] * (-10.0 / (81.0 * user_ptr->mu * user_ptr->mu) + 2.0 * 292.0 / (2187.0 * user_ptr->mu * user_ptr->mu * user_ptr->mu)) + x_ptr[0];
  PetscCall(VecRestoreArray(user_ptr->Mup[0], &x_ptr));
  PetscCall(VecRestoreArrayRead(user_ptr->Lambda[0], &y_ptr));
  PetscCall(VecRestoreArray(G, &g));
  PetscFunctionReturn(0);
}

PetscErrorCode FormHessian(Tao tao, Vec P, Mat H, Mat Hpre, void *ctx)
{
  User           user_ptr = (User)ctx;
  PetscScalar    harr[1];
  const PetscInt rows[1] = {0};
  PetscInt       col     = 0;

  PetscFunctionBeginUser;
  PetscCall(Adjoint2(P, harr, user_ptr));
  PetscCall(MatSetValues(H, 1, rows, 1, &col, harr, INSERT_VALUES));

  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  if (H != Hpre) {
    PetscCall(MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode Adjoint2(Vec P, PetscScalar arr[], User ctx)
{
  TS                 ts = ctx->ts;
  const PetscScalar *z_ptr;
  PetscScalar       *x_ptr, *y_ptr, dzdp, dzdp2;
  Mat                tlmsen;

  PetscFunctionBeginUser;
  /* Reset TSAdjoint so that AdjointSetUp will be called again */
  PetscCall(TSAdjointReset(ts));

  /* The directional vector should be 1 since it is one-dimensional */
  PetscCall(VecGetArray(ctx->Dir, &x_ptr));
  x_ptr[0] = 1.;
  PetscCall(VecRestoreArray(ctx->Dir, &x_ptr));

  PetscCall(VecGetArrayRead(P, &z_ptr));
  ctx->mu = z_ptr[0];
  PetscCall(VecRestoreArrayRead(P, &z_ptr));

  dzdp  = -10.0 / (81.0 * ctx->mu * ctx->mu) + 2.0 * 292.0 / (2187.0 * ctx->mu * ctx->mu * ctx->mu);
  dzdp2 = 2. * 10.0 / (81.0 * ctx->mu * ctx->mu * ctx->mu) - 3.0 * 2.0 * 292.0 / (2187.0 * ctx->mu * ctx->mu * ctx->mu * ctx->mu);

  PetscCall(TSSetTime(ts, 0.0));
  PetscCall(TSSetStepNumber(ts, 0));
  PetscCall(TSSetTimeStep(ts, PetscRealConstant(0.001))); /* can be overwritten by command line options */
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetCostHessianProducts(ts, 1, ctx->Lambda2, ctx->Mup2, ctx->Dir));

  PetscCall(MatZeroEntries(ctx->Jacp));
  PetscCall(MatSetValue(ctx->Jacp, 1, 0, dzdp, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(ctx->Jacp, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(ctx->Jacp, MAT_FINAL_ASSEMBLY));

  PetscCall(TSAdjointSetForward(ts, ctx->Jacp));
  PetscCall(VecGetArray(ctx->U, &y_ptr));
  y_ptr[0] = 2.0;
  y_ptr[1] = -2.0 / 3.0 + 10.0 / (81.0 * ctx->mu) - 292.0 / (2187.0 * ctx->mu * ctx->mu);
  PetscCall(VecRestoreArray(ctx->U, &y_ptr));
  PetscCall(TSSolve(ts, ctx->U));

  /* Set terminal conditions for first- and second-order adjonts */
  PetscCall(VecGetArrayRead(ctx->U, &z_ptr));
  PetscCall(VecGetArray(ctx->Lambda[0], &y_ptr));
  y_ptr[0] = 2. * (z_ptr[0] - ctx->ob[0]);
  y_ptr[1] = 2. * (z_ptr[1] - ctx->ob[1]);
  PetscCall(VecRestoreArray(ctx->Lambda[0], &y_ptr));
  PetscCall(VecRestoreArrayRead(ctx->U, &z_ptr));
  PetscCall(VecGetArray(ctx->Mup[0], &y_ptr));
  y_ptr[0] = 0.0;
  PetscCall(VecRestoreArray(ctx->Mup[0], &y_ptr));
  PetscCall(TSForwardGetSensitivities(ts, NULL, &tlmsen));
  PetscCall(MatDenseGetColumn(tlmsen, 0, &x_ptr));
  PetscCall(VecGetArray(ctx->Lambda2[0], &y_ptr));
  y_ptr[0] = 2. * x_ptr[0];
  y_ptr[1] = 2. * x_ptr[1];
  PetscCall(VecRestoreArray(ctx->Lambda2[0], &y_ptr));
  PetscCall(VecGetArray(ctx->Mup2[0], &y_ptr));
  y_ptr[0] = 0.0;
  PetscCall(VecRestoreArray(ctx->Mup2[0], &y_ptr));
  PetscCall(MatDenseRestoreColumn(tlmsen, &x_ptr));
  PetscCall(TSSetCostGradients(ts, 1, ctx->Lambda, ctx->Mup));
  if (ctx->implicitform) {
    PetscCall(TSSetIHessianProduct(ts, ctx->Ihp1, IHessianProductUU, ctx->Ihp2, IHessianProductUP, ctx->Ihp3, IHessianProductPU, ctx->Ihp4, IHessianProductPP, ctx));
  } else {
    PetscCall(TSSetRHSHessianProduct(ts, ctx->Ihp1, RHSHessianProductUU, ctx->Ihp2, RHSHessianProductUP, ctx->Ihp3, RHSHessianProductPU, ctx->Ihp4, RHSHessianProductPP, ctx));
  }
  PetscCall(TSAdjointSolve(ts));

  PetscCall(VecGetArray(ctx->Lambda[0], &x_ptr));
  PetscCall(VecGetArray(ctx->Lambda2[0], &y_ptr));
  PetscCall(VecGetArrayRead(ctx->Mup2[0], &z_ptr));

  arr[0] = x_ptr[1] * dzdp2 + y_ptr[1] * dzdp2 + z_ptr[0];

  PetscCall(VecRestoreArray(ctx->Lambda2[0], &x_ptr));
  PetscCall(VecRestoreArray(ctx->Lambda2[0], &y_ptr));
  PetscCall(VecRestoreArrayRead(ctx->Mup2[0], &z_ptr));

  /* Disable second-order adjoint mode */
  PetscCall(TSAdjointReset(ts));
  PetscCall(TSAdjointResetForward(ts));
  PetscFunctionReturn(0);
}

/*TEST
    build:
      requires: !complex !single
    test:
      args:  -implicitform 0 -ts_type rk -ts_adapt_type none -mu 10 -ts_dt 0.1 -viewer_binary_skip_info -tao_monitor -tao_view
      output_file: output/ex20opt_p_1.out

    test:
      suffix: 2
      args:  -implicitform 0 -ts_type rk -ts_adapt_type none -mu 10 -ts_dt 0.01 -viewer_binary_skip_info -tao_monitor -tao_type bntr -tao_bnk_pc_type none
      output_file: output/ex20opt_p_2.out

    test:
      suffix: 3
      args:  -ts_type cn -ts_adapt_type none -mu 100 -ts_dt 0.01 -viewer_binary_skip_info -tao_monitor -tao_view
      output_file: output/ex20opt_p_3.out

    test:
      suffix: 4
      args:  -ts_type cn -ts_adapt_type none -mu 100 -ts_dt 0.01 -viewer_binary_skip_info -tao_monitor -tao_type bntr -tao_bnk_pc_type none
      output_file: output/ex20opt_p_4.out

TEST*/
