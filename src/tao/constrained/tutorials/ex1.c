/* Program usage: mpiexec -n 2 ./ex1 [-help] [all TAO options] */

/* ----------------------------------------------------------------------
min f(x) = (x0 - 2)^2 + (x1 - 2)^2 - 2*(x0 + x1)
s.t.  x0^2 + x1 - 2 = 0
      0  <= x0^2 - x1 <= 1
      -1 <= x0, x1 <= 2
-->
      g(x)  = 0
      h(x) >= 0
      -1 <= x0, x1 <= 2
where
      g(x) = x0^2 + x1 - 2
      h(x) = [x0^2 - x1
              1 -(x0^2 - x1)]
---------------------------------------------------------------------- */

#include <petsctao.h>

static char help[] = "Solves constrained optimization problem using pdipm.\n\
Input parameters include:\n\
  -tao_type pdipm    : sets Tao solver\n\
  -no_eq             : removes the equaility constraints from the problem\n\
  -init_view         : view initial object setup\n\
  -snes_fd           : snes with finite difference Jacobian (needed for pdipm)\n\
  -snes_compare_explicit : compare user Jacobian with finite difference Jacobian \n\
  -tao_cmonitor      : convergence monitor with constraint norm \n\
  -tao_view_solution : view exact solution at each iteration\n\
  Note: external package MUMPS is required to run pdipm in parallel. This is designed for a maximum of 2 processors, the code will error if size > 2.\n";

/*
   User-defined application context - contains data needed by the application
*/
typedef struct {
  PetscInt   n;  /* Global length of x */
  PetscInt   ne; /* Global number of equality constraints */
  PetscInt   ni; /* Global number of inequality constraints */
  PetscBool  noeqflag, initview;
  Vec        x, xl, xu;
  Vec        ce, ci, bl, bu, Xseq;
  Mat        Ae, Ai, H;
  VecScatter scat;
} AppCtx;

/* -------- User-defined Routines --------- */
PetscErrorCode InitializeProblem(AppCtx *);
PetscErrorCode DestroyProblem(AppCtx *);
PetscErrorCode FormFunctionGradient(Tao, Vec, PetscReal *, Vec, void *);
PetscErrorCode FormPDIPMHessian(Tao, Vec, Mat, Mat, void *);
PetscErrorCode FormInequalityConstraints(Tao, Vec, Vec, void *);
PetscErrorCode FormEqualityConstraints(Tao, Vec, Vec, void *);
PetscErrorCode FormInequalityJacobian(Tao, Vec, Mat, Mat, void *);
PetscErrorCode FormEqualityJacobian(Tao, Vec, Mat, Mat, void *);

int main(int argc, char **argv)
{
  Tao         tao;
  KSP         ksp;
  PC          pc;
  AppCtx      user; /* application context */
  Vec         x, G, CI, CE;
  PetscMPIInt size;
  TaoType     type;
  PetscReal   f;
  PetscBool   pdipm, mumps;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size <= 2, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "More than 2 processors detected. Example written to use max of 2 processors.");

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "---- Constrained Problem -----\n"));
  PetscCall(InitializeProblem(&user)); /* sets up problem, function below */

  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetType(tao, TAOALMM));
  PetscCall(TaoSetSolution(tao, user.x));
  PetscCall(TaoSetVariableBounds(tao, user.xl, user.xu));
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormFunctionGradient, (void *)&user));
  PetscCall(TaoSetTolerances(tao, 1.e-4, 0.0, 0.0));
  PetscCall(TaoSetConstraintTolerances(tao, 1.e-4, 0.0));
  PetscCall(TaoSetMaximumFunctionEvaluations(tao, 1e6));
  PetscCall(TaoSetFromOptions(tao));

  if (!user.noeqflag) {
    PetscCall(TaoSetEqualityConstraintsRoutine(tao, user.ce, FormEqualityConstraints, (void *)&user));
    PetscCall(TaoSetJacobianEqualityRoutine(tao, user.Ae, user.Ae, FormEqualityJacobian, (void *)&user));
  }
  PetscCall(TaoSetInequalityConstraintsRoutine(tao, user.ci, FormInequalityConstraints, (void *)&user));
  PetscCall(TaoSetJacobianInequalityRoutine(tao, user.Ai, user.Ai, FormInequalityJacobian, (void *)&user));

  PetscCall(TaoGetType(tao, &type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOPDIPM, &pdipm));
  if (pdipm) {
    /*
      PDIPM produces an indefinite KKT matrix with some zeros along the diagonal
      Inverting this indefinite matrix requires PETSc to be configured with MUMPS
    */
    PetscCall(PetscHasExternalPackage("mumps", &mumps));
    PetscCheck(mumps, PetscObjectComm((PetscObject)tao), PETSC_ERR_SUP, "TAOPDIPM requires PETSc to be configured with MUMPS (--download-mumps)");
    PetscCall(TaoGetKSP(tao, &ksp));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
    PetscCall(KSPSetType(ksp, KSPPREONLY));
    PetscCall(PCSetType(pc, PCCHOLESKY));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(TaoSetHessian(tao, user.H, user.H, FormPDIPMHessian, (void *)&user));
  }

  /* Print out an initial view of the problem */
  if (user.initview) {
    PetscCall(TaoSetUp(tao));
    PetscCall(VecDuplicate(user.x, &G));
    PetscCall(FormFunctionGradient(tao, user.x, &f, G, (void *)&user));
    PetscCall(FormPDIPMHessian(tao, user.x, user.H, user.H, (void *)&user));
    PetscCall(PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nInitial point X:\n"));
    PetscCall(VecView(user.x, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nInitial objective f(x) = %g\n", (double)f));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nInitial gradient and Hessian:\n"));
    PetscCall(VecView(G, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(MatView(user.H, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(VecDestroy(&G));
    PetscCall(FormInequalityJacobian(tao, user.x, user.Ai, user.Ai, (void *)&user));
    PetscCall(MatCreateVecs(user.Ai, NULL, &CI));
    PetscCall(FormInequalityConstraints(tao, user.x, CI, (void *)&user));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nInitial inequality constraints and Jacobian:\n"));
    PetscCall(VecView(CI, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(MatView(user.Ai, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(VecDestroy(&CI));
    if (!user.noeqflag) {
      PetscCall(FormEqualityJacobian(tao, user.x, user.Ae, user.Ae, (void *)&user));
      PetscCall(MatCreateVecs(user.Ae, NULL, &CE));
      PetscCall(FormEqualityConstraints(tao, user.x, CE, (void *)&user));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nInitial equality constraints and Jacobian:\n"));
      PetscCall(VecView(CE, PETSC_VIEWER_STDOUT_WORLD));
      PetscCall(MatView(user.Ae, PETSC_VIEWER_STDOUT_WORLD));
      PetscCall(VecDestroy(&CE));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
    PetscCall(PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_WORLD));
  }

  PetscCall(TaoSolve(tao));
  PetscCall(TaoGetSolution(tao, &x));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Found solution:\n"));
  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));

  /* Free objects */
  PetscCall(DestroyProblem(&user));
  PetscCall(TaoDestroy(&tao));
  PetscCall(PetscFinalize());
  return PETSC_SUCCESS;
}

PetscErrorCode InitializeProblem(AppCtx *user)
{
  PetscMPIInt size;
  PetscMPIInt rank;
  PetscInt    nloc, neloc, niloc;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  user->noeqflag = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-no_eq", &user->noeqflag, NULL));
  user->initview = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-init_view", &user->initview, NULL));

  /* Tell user the correct solution, not an error checking */
  if (!user->noeqflag) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Expected solution: f(1, 1) = -2\n"));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Expected solution (-no_eq): f(1.73205, 2) = -7.3923\n"));
  }

  /* create vector x and set initial values */
  user->n = 2; /* global length */
  nloc    = (size == 1) ? user->n : 1;
  PetscCall(VecCreate(PETSC_COMM_WORLD, &user->x));
  PetscCall(VecSetSizes(user->x, nloc, user->n));
  PetscCall(VecSetFromOptions(user->x));
  PetscCall(VecSet(user->x, 0.0));

  /* create and set lower and upper bound vectors */
  PetscCall(VecDuplicate(user->x, &user->xl));
  PetscCall(VecDuplicate(user->x, &user->xu));
  PetscCall(VecSet(user->xl, -1.0));
  PetscCall(VecSet(user->xu, 2.0));

  /* create scater to zero */
  PetscCall(VecScatterCreateToZero(user->x, &user->scat, &user->Xseq));

  user->ne = 1;
  user->ni = 2;
  neloc    = (rank == 0) ? user->ne : 0;
  niloc    = (size == 1) ? user->ni : 1;

  if (!user->noeqflag) {
    PetscCall(VecCreate(PETSC_COMM_WORLD, &user->ce)); /* a 1x1 vec for equality constraints */
    PetscCall(VecSetSizes(user->ce, neloc, user->ne));
    PetscCall(VecSetFromOptions(user->ce));
    PetscCall(VecSetUp(user->ce));
  }

  PetscCall(VecCreate(PETSC_COMM_WORLD, &user->ci)); /* a 2x1 vec for inequality constraints */
  PetscCall(VecSetSizes(user->ci, niloc, user->ni));
  PetscCall(VecSetFromOptions(user->ci));
  PetscCall(VecSetUp(user->ci));

  /* nexn & nixn matricies for equally and inequalty constraints */
  if (!user->noeqflag) {
    PetscCall(MatCreate(PETSC_COMM_WORLD, &user->Ae));
    PetscCall(MatSetSizes(user->Ae, neloc, nloc, user->ne, user->n));
    PetscCall(MatSetFromOptions(user->Ae));
    PetscCall(MatSetUp(user->Ae));
  }

  PetscCall(MatCreate(PETSC_COMM_WORLD, &user->Ai));
  PetscCall(MatSetSizes(user->Ai, niloc, nloc, user->ni, user->n));
  PetscCall(MatSetFromOptions(user->Ai));
  PetscCall(MatSetUp(user->Ai));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &user->H));
  PetscCall(MatSetSizes(user->H, nloc, nloc, user->n, user->n));
  PetscCall(MatSetFromOptions(user->H));
  PetscCall(MatSetUp(user->H));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DestroyProblem(AppCtx *user)
{
  PetscFunctionBegin;
  if (!user->noeqflag) PetscCall(MatDestroy(&user->Ae));
  PetscCall(MatDestroy(&user->Ai));
  PetscCall(MatDestroy(&user->H));

  PetscCall(VecDestroy(&user->x));
  if (!user->noeqflag) PetscCall(VecDestroy(&user->ce));
  PetscCall(VecDestroy(&user->ci));
  PetscCall(VecDestroy(&user->xl));
  PetscCall(VecDestroy(&user->xu));
  PetscCall(VecDestroy(&user->Xseq));
  PetscCall(VecScatterDestroy(&user->scat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Evaluate
   f(x) = (x0 - 2)^2 + (x1 - 2)^2 - 2*(x0 + x1)
   G = grad f = [2*(x0 - 2) - 2, 2*(x1 - 2) - 2] = 2*X - 6
*/
PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ctx)
{
  const PetscScalar *x;
  MPI_Comm           comm;
  PetscMPIInt        rank;
  PetscReal          fin;
  AppCtx            *user = (AppCtx *)ctx;
  Vec                Xseq = user->Xseq;
  VecScatter         scat = user->scat;

  PetscFunctionBegin;
  /* f = (x0 - 2)^2 + (x1 - 2)^2 - 2*(x0 + x1) */
  PetscCall(PetscObjectGetComm((PetscObject)tao, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(VecScatterBegin(scat, X, Xseq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scat, X, Xseq, INSERT_VALUES, SCATTER_FORWARD));

  fin = 0.0;
  if (rank == 0) {
    PetscCall(VecGetArrayRead(Xseq, &x));
    fin = (x[0] - 2.0) * (x[0] - 2.0) + (x[1] - 2.0) * (x[1] - 2.0) - 2.0 * (x[0] + x[1]);
    PetscCall(VecRestoreArrayRead(Xseq, &x));
  }
  PetscCall(MPIU_Allreduce(&fin, f, 1, MPIU_REAL, MPIU_SUM, comm));

  /* G = 2*X - 6 */
  PetscCall(VecSet(G, -6.0));
  PetscCall(VecAXPY(G, 2.0, X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Evaluate PDIPM Hessian, see Eqn(22) in http://doi.org/10.1049/gtd2.12708
   H = Wxx = fxx        + Jacobian(grad g^T*DE)   - Jacobian(grad h^T*DI)]
           = fxx        + Jacobin([2*x0; 1]de[0]) - Jacobian([2*x0, -2*x0; -1, 1][di[0] di[1]]^T)
           = [2 0; 0 2] + [2*de[0]  0;      0  0] - [2*di[0]-2*di[1], 0; 0, 0]
           = [ 2*(1+de[0]-di[0]+di[1]), 0;
                          0,            2]
*/
PetscErrorCode FormPDIPMHessian(Tao tao, Vec x, Mat H, Mat Hpre, void *ctx)
{
  AppCtx            *user = (AppCtx *)ctx;
  Vec                DE, DI;
  const PetscScalar *de, *di;
  PetscInt           zero = 0, one = 1;
  PetscScalar        two = 2.0;
  PetscScalar        val = 0.0;
  Vec                Deseq, Diseq;
  VecScatter         Descat, Discat;
  PetscMPIInt        rank;
  MPI_Comm           comm;

  PetscFunctionBegin;
  PetscCall(TaoGetDualVariables(tao, &DE, &DI));

  PetscCall(PetscObjectGetComm((PetscObject)tao, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  if (!user->noeqflag) {
    PetscCall(VecScatterCreateToZero(DE, &Descat, &Deseq));
    PetscCall(VecScatterBegin(Descat, DE, Deseq, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(Descat, DE, Deseq, INSERT_VALUES, SCATTER_FORWARD));
  }
  PetscCall(VecScatterCreateToZero(DI, &Discat, &Diseq));
  PetscCall(VecScatterBegin(Discat, DI, Diseq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(Discat, DI, Diseq, INSERT_VALUES, SCATTER_FORWARD));

  if (rank == 0) {
    if (!user->noeqflag) { PetscCall(VecGetArrayRead(Deseq, &de)); /* places equality constraint dual into array */ }
    PetscCall(VecGetArrayRead(Diseq, &di)); /* places inequality constraint dual into array */

    if (!user->noeqflag) {
      val = 2.0 * (1 + de[0] - di[0] + di[1]);
      PetscCall(VecRestoreArrayRead(Deseq, &de));
      PetscCall(VecRestoreArrayRead(Diseq, &di));
    } else {
      val = 2.0 * (1 - di[0] + di[1]);
    }
    PetscCall(VecRestoreArrayRead(Diseq, &di));
    PetscCall(MatSetValues(H, 1, &zero, 1, &zero, &val, INSERT_VALUES));
    PetscCall(MatSetValues(H, 1, &one, 1, &one, &two, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  if (!user->noeqflag) {
    PetscCall(VecScatterDestroy(&Descat));
    PetscCall(VecDestroy(&Deseq));
  }
  PetscCall(VecScatterDestroy(&Discat));
  PetscCall(VecDestroy(&Diseq));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Evaluate
   h = [ x0^2 - x1;
         1 -(x0^2 - x1)]
*/
PetscErrorCode FormInequalityConstraints(Tao tao, Vec X, Vec CI, void *ctx)
{
  const PetscScalar *x;
  PetscScalar        ci;
  MPI_Comm           comm;
  PetscMPIInt        rank;
  AppCtx            *user = (AppCtx *)ctx;
  Vec                Xseq = user->Xseq;
  VecScatter         scat = user->scat;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)tao, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(VecScatterBegin(scat, X, Xseq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scat, X, Xseq, INSERT_VALUES, SCATTER_FORWARD));

  if (rank == 0) {
    PetscCall(VecGetArrayRead(Xseq, &x));
    ci = x[0] * x[0] - x[1];
    PetscCall(VecSetValue(CI, 0, ci, INSERT_VALUES));
    ci = -x[0] * x[0] + x[1] + 1.0;
    PetscCall(VecSetValue(CI, 1, ci, INSERT_VALUES));
    PetscCall(VecRestoreArrayRead(Xseq, &x));
  }
  PetscCall(VecAssemblyBegin(CI));
  PetscCall(VecAssemblyEnd(CI));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Evaluate
   g = [ x0^2 + x1 - 2]
*/
PetscErrorCode FormEqualityConstraints(Tao tao, Vec X, Vec CE, void *ctx)
{
  const PetscScalar *x;
  PetscScalar        ce;
  MPI_Comm           comm;
  PetscMPIInt        rank;
  AppCtx            *user = (AppCtx *)ctx;
  Vec                Xseq = user->Xseq;
  VecScatter         scat = user->scat;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)tao, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(VecScatterBegin(scat, X, Xseq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scat, X, Xseq, INSERT_VALUES, SCATTER_FORWARD));

  if (rank == 0) {
    PetscCall(VecGetArrayRead(Xseq, &x));
    ce = x[0] * x[0] + x[1] - 2.0;
    PetscCall(VecSetValue(CE, 0, ce, INSERT_VALUES));
    PetscCall(VecRestoreArrayRead(Xseq, &x));
  }
  PetscCall(VecAssemblyBegin(CE));
  PetscCall(VecAssemblyEnd(CE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  grad h = [  2*x0, -1;
             -2*x0,  1]
*/
PetscErrorCode FormInequalityJacobian(Tao tao, Vec X, Mat JI, Mat JIpre, void *ctx)
{
  AppCtx            *user = (AppCtx *)ctx;
  PetscInt           zero = 0, one = 1, cols[2];
  PetscScalar        vals[2];
  const PetscScalar *x;
  Vec                Xseq = user->Xseq;
  VecScatter         scat = user->scat;
  MPI_Comm           comm;
  PetscMPIInt        rank;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)tao, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(VecScatterBegin(scat, X, Xseq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scat, X, Xseq, INSERT_VALUES, SCATTER_FORWARD));

  PetscCall(VecGetArrayRead(Xseq, &x));
  if (rank == 0) {
    cols[0] = 0;
    cols[1] = 1;
    vals[0] = 2 * x[0];
    vals[1] = -1.0;
    PetscCall(MatSetValues(JI, 1, &zero, 2, cols, vals, INSERT_VALUES));
    vals[0] = -2 * x[0];
    vals[1] = 1.0;
    PetscCall(MatSetValues(JI, 1, &one, 2, cols, vals, INSERT_VALUES));
  }
  PetscCall(VecRestoreArrayRead(Xseq, &x));
  PetscCall(MatAssemblyBegin(JI, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(JI, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  grad g = [2*x0, 1.0]
*/
PetscErrorCode FormEqualityJacobian(Tao tao, Vec X, Mat JE, Mat JEpre, void *ctx)
{
  PetscInt           zero = 0, cols[2];
  PetscScalar        vals[2];
  const PetscScalar *x;
  PetscMPIInt        rank;
  MPI_Comm           comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)tao, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  if (rank == 0) {
    PetscCall(VecGetArrayRead(X, &x));
    cols[0] = 0;
    cols[1] = 1;
    vals[0] = 2 * x[0];
    vals[1] = 1.0;
    PetscCall(MatSetValues(JE, 1, &zero, 2, cols, vals, INSERT_VALUES));
    PetscCall(VecRestoreArrayRead(X, &x));
  }
  PetscCall(MatAssemblyBegin(JE, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(JE, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   build:
      requires: !complex !defined(PETSC_USE_CXX)

   test:
      args: -tao_converged_reason -tao_gatol 1.e-6 -tao_type pdipm -tao_pdipm_kkt_shift_pd
      requires: mumps
      filter: sed  -e "s/CONVERGED_GATOL iterations *[0-9]\{1,\}/CONVERGED_GATOL/g"

   test:
      suffix: pdipm_2
      requires: mumps
      nsize: 2
      args: -tao_converged_reason -tao_gatol 1.e-6 -tao_type pdipm
      filter: sed  -e "s/CONVERGED_GATOL iterations *[0-9]\{1,\}/CONVERGED_GATOL/g"

   test:
      suffix: 2
      args: -tao_converged_reason
      filter: sed  -e "s/CONVERGED_GATOL iterations *[0-9]\{1,\}/CONVERGED_GATOL/g"

   test:
      suffix: 3
      args: -tao_converged_reason -no_eq
      filter: sed  -e "s/CONVERGED_GATOL iterations *[0-9]\{1,\}/CONVERGED_GATOL/g"

   test:
      suffix: 4
      args: -tao_converged_reason -tao_almm_type classic
      requires: !single
      filter: sed  -e "s/CONVERGED_GATOL iterations *[0-9]\{1,\}/CONVERGED_GATOL/g"

   test:
      suffix: 5
      args: -tao_converged_reason -tao_almm_type classic -no_eq
      requires: !single !defined(PETSCTEST_VALGRIND)
      filter: sed  -e "s/CONVERGED_GATOL iterations *[0-9]\{1,\}/CONVERGED_GATOL/g"

   test:
      suffix: 6
      args: -tao_converged_reason -tao_almm_subsolver_tao_type bqnktr
      filter: sed  -e "s/CONVERGED_GATOL iterations *[0-9]\{1,\}/CONVERGED_GATOL/g"

   test:
      suffix: 7
      args: -tao_converged_reason -tao_almm_subsolver_tao_type bncg
      filter: sed  -e "s/CONVERGED_GATOL iterations *[0-9]\{1,\}/CONVERGED_GATOL/g"

   test:
      suffix: 8
      nsize: 2
      args: -tao_converged_reason
      filter: sed  -e "s/CONVERGED_GATOL iterations *[0-9]\{1,\}/CONVERGED_GATOL/g"

   test:
      suffix: 9
      nsize: 2
      args: -tao_converged_reason -vec_type cuda -mat_type aijcusparse
      requires: cuda
      filter: sed  -e "s/CONVERGED_GATOL iterations *[0-9]\{1,\}/CONVERGED_GATOL/g"

TEST*/
