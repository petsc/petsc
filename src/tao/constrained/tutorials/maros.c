/* Program usage: mpiexec -n 1 maros1 [-help] [all TAO options] */

/* ----------------------------------------------------------------------
TODO Explain maros example
---------------------------------------------------------------------- */

#include <petsctao.h>

static char help[] = "";

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines, FormFunction(),
   FormGradient(), and FormHessian().
*/

/*
   x,d in R^n
   f in R
   bin in R^mi
   beq in R^me
   Aeq in R^(me x n)
   Ain in R^(mi x n)
   H in R^(n x n)
   min f=(1/2)*x'*H*x + d'*x
   s.t.  Aeq*x == beq
         Ain*x >= bin
*/
typedef struct {
  char     name[32];
  PetscInt n;  /* Length x */
  PetscInt me; /* number of equality constraints */
  PetscInt mi; /* number of inequality constraints */
  PetscInt m;  /* me+mi */
  Mat      Aeq, Ain, H;
  Vec      beq, bin, d;
} AppCtx;

/* -------- User-defined Routines --------- */

PetscErrorCode InitializeProblem(AppCtx *);
PetscErrorCode DestroyProblem(AppCtx *);
PetscErrorCode FormFunctionGradient(Tao, Vec, PetscReal *, Vec, void *);
PetscErrorCode FormHessian(Tao, Vec, Mat, Mat, void *);
PetscErrorCode FormInequalityConstraints(Tao, Vec, Vec, void *);
PetscErrorCode FormEqualityConstraints(Tao, Vec, Vec, void *);
PetscErrorCode FormInequalityJacobian(Tao, Vec, Mat, Mat, void *);
PetscErrorCode FormEqualityJacobian(Tao, Vec, Mat, Mat, void *);

PetscErrorCode main(int argc, char **argv)
{
  PetscMPIInt        size;
  Vec                x; /* solution */
  KSP                ksp;
  PC                 pc;
  Vec                ceq, cin;
  PetscBool          flg; /* A return value when checking for use options */
  Tao                tao; /* Tao solver context */
  TaoConvergedReason reason;
  AppCtx             user; /* application context */

  /* Initialize TAO,PETSc */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  /* Specify default parameters for the problem, check for command-line overrides */
  PetscCall(PetscStrncpy(user.name, "HS21", sizeof(user.name)));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-cutername", user.name, sizeof(user.name), &flg));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n---- MAROS Problem %s -----\n", user.name));
  PetscCall(InitializeProblem(&user));
  PetscCall(VecDuplicate(user.d, &x));
  PetscCall(VecDuplicate(user.beq, &ceq));
  PetscCall(VecDuplicate(user.bin, &cin));
  PetscCall(VecSet(x, 1.0));

  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetType(tao, TAOIPM));
  PetscCall(TaoSetSolution(tao, x));
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormFunctionGradient, (void *)&user));
  PetscCall(TaoSetEqualityConstraintsRoutine(tao, ceq, FormEqualityConstraints, (void *)&user));
  PetscCall(TaoSetInequalityConstraintsRoutine(tao, cin, FormInequalityConstraints, (void *)&user));
  PetscCall(TaoSetInequalityBounds(tao, user.bin, NULL));
  PetscCall(TaoSetJacobianEqualityRoutine(tao, user.Aeq, user.Aeq, FormEqualityJacobian, (void *)&user));
  PetscCall(TaoSetJacobianInequalityRoutine(tao, user.Ain, user.Ain, FormInequalityJacobian, (void *)&user));
  PetscCall(TaoSetHessian(tao, user.H, user.H, FormHessian, (void *)&user));
  PetscCall(TaoGetKSP(tao, &ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCLU));
  /*
      This algorithm produces matrices with zeros along the diagonal therefore we need to use
    SuperLU which does partial pivoting
  */
  PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERSUPERLU));
  PetscCall(KSPSetType(ksp, KSPPREONLY));
  PetscCall(TaoSetTolerances(tao, 0, 0, 0));

  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));
  PetscCall(TaoGetConvergedReason(tao, &reason));
  if (reason < 0) {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "TAO failed to converge due to %s.\n", TaoConvergedReasons[reason]));
  } else {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "Optimization completed with status %s.\n", TaoConvergedReasons[reason]));
  }

  PetscCall(DestroyProblem(&user));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&ceq));
  PetscCall(VecDestroy(&cin));
  PetscCall(TaoDestroy(&tao));

  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode InitializeProblem(AppCtx *user)
{
  PetscViewer loader;
  MPI_Comm    comm;
  PetscInt    nrows, ncols, i;
  PetscScalar one = 1.0;
  char        filebase[128];
  char        filename[128];

  PetscFunctionBegin;
  comm = PETSC_COMM_WORLD;
  PetscCall(PetscStrncpy(filebase, user->name, sizeof(filebase)));
  PetscCall(PetscStrlcat(filebase, "/", sizeof(filebase)));
  PetscCall(PetscStrncpy(filename, filebase, sizeof(filename)));
  PetscCall(PetscStrlcat(filename, "f", sizeof(filename)));
  PetscCall(PetscViewerBinaryOpen(comm, filename, FILE_MODE_READ, &loader));

  PetscCall(VecCreate(comm, &user->d));
  PetscCall(VecLoad(user->d, loader));
  PetscCall(PetscViewerDestroy(&loader));
  PetscCall(VecGetSize(user->d, &nrows));
  PetscCall(VecSetFromOptions(user->d));
  user->n = nrows;

  PetscCall(PetscStrncpy(filename, filebase, sizeof(filename)));
  PetscCall(PetscStrlcat(filename, "H", sizeof(filename)));
  PetscCall(PetscViewerBinaryOpen(comm, filename, FILE_MODE_READ, &loader));

  PetscCall(MatCreate(comm, &user->H));
  PetscCall(MatSetSizes(user->H, PETSC_DECIDE, PETSC_DECIDE, nrows, nrows));
  PetscCall(MatLoad(user->H, loader));
  PetscCall(PetscViewerDestroy(&loader));
  PetscCall(MatGetSize(user->H, &nrows, &ncols));
  PetscCheck(nrows == user->n, comm, PETSC_ERR_ARG_SIZ, "H: nrows != n");
  PetscCheck(ncols == user->n, comm, PETSC_ERR_ARG_SIZ, "H: ncols != n");
  PetscCall(MatSetFromOptions(user->H));

  PetscCall(PetscStrncpy(filename, filebase, sizeof(filename)));
  PetscCall(PetscStrlcat(filename, "Aeq", sizeof(filename)));
  PetscCall(PetscViewerBinaryOpen(comm, filename, FILE_MODE_READ, &loader));
  PetscCall(MatCreate(comm, &user->Aeq));
  PetscCall(MatLoad(user->Aeq, loader));
  PetscCall(PetscViewerDestroy(&loader));
  PetscCall(MatGetSize(user->Aeq, &nrows, &ncols));
  PetscCheck(ncols == user->n, comm, PETSC_ERR_ARG_SIZ, "Aeq ncols != H nrows");
  PetscCall(MatSetFromOptions(user->Aeq));
  user->me = nrows;

  PetscCall(PetscStrncpy(filename, filebase, sizeof(filename)));
  PetscCall(PetscStrlcat(filename, "Beq", sizeof(filename)));
  PetscCall(PetscViewerBinaryOpen(comm, filename, FILE_MODE_READ, &loader));
  PetscCall(VecCreate(comm, &user->beq));
  PetscCall(VecLoad(user->beq, loader));
  PetscCall(PetscViewerDestroy(&loader));
  PetscCall(VecGetSize(user->beq, &nrows));
  PetscCheck(nrows == user->me, comm, PETSC_ERR_ARG_SIZ, "Aeq nrows != Beq n");
  PetscCall(VecSetFromOptions(user->beq));

  user->mi = user->n;
  /* Ain = eye(n,n) */
  PetscCall(MatCreate(comm, &user->Ain));
  PetscCall(MatSetType(user->Ain, MATAIJ));
  PetscCall(MatSetSizes(user->Ain, PETSC_DECIDE, PETSC_DECIDE, user->mi, user->mi));

  PetscCall(MatMPIAIJSetPreallocation(user->Ain, 1, NULL, 0, NULL));
  PetscCall(MatSeqAIJSetPreallocation(user->Ain, 1, NULL));

  for (i = 0; i < user->mi; i++) PetscCall(MatSetValues(user->Ain, 1, &i, 1, &i, &one, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(user->Ain, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(user->Ain, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetFromOptions(user->Ain));

  /* bin = [0,0 ... 0]' */
  PetscCall(VecCreate(comm, &user->bin));
  PetscCall(VecSetType(user->bin, VECMPI));
  PetscCall(VecSetSizes(user->bin, PETSC_DECIDE, user->mi));
  PetscCall(VecSet(user->bin, 0.0));
  PetscCall(VecSetFromOptions(user->bin));
  user->m = user->me + user->mi;
  PetscFunctionReturn(0);
}

PetscErrorCode DestroyProblem(AppCtx *user)
{
  PetscFunctionBegin;
  PetscCall(MatDestroy(&user->H));
  PetscCall(MatDestroy(&user->Aeq));
  PetscCall(MatDestroy(&user->Ain));
  PetscCall(VecDestroy(&user->beq));
  PetscCall(VecDestroy(&user->bin));
  PetscCall(VecDestroy(&user->d));
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunctionGradient(Tao tao, Vec x, PetscReal *f, Vec g, void *ctx)
{
  AppCtx     *user = (AppCtx *)ctx;
  PetscScalar xtHx;

  PetscFunctionBegin;
  PetscCall(MatMult(user->H, x, g));
  PetscCall(VecDot(x, g, &xtHx));
  PetscCall(VecDot(x, user->d, f));
  *f += 0.5 * xtHx;
  PetscCall(VecAXPY(g, 1.0, user->d));
  PetscFunctionReturn(0);
}

PetscErrorCode FormHessian(Tao tao, Vec x, Mat H, Mat Hpre, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode FormInequalityConstraints(Tao tao, Vec x, Vec ci, void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscFunctionBegin;
  PetscCall(MatMult(user->Ain, x, ci));
  PetscFunctionReturn(0);
}

PetscErrorCode FormEqualityConstraints(Tao tao, Vec x, Vec ce, void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscFunctionBegin;
  PetscCall(MatMult(user->Aeq, x, ce));
  PetscCall(VecAXPY(ce, -1.0, user->beq));
  PetscFunctionReturn(0);
}

PetscErrorCode FormInequalityJacobian(Tao tao, Vec x, Mat JI, Mat JIpre, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode FormEqualityJacobian(Tao tao, Vec x, Mat JE, Mat JEpre, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex

   test:
      requires: superlu
      localrunfiles: HS21

TEST*/
