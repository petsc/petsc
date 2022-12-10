
static char help[] = "Tests the multigrid code.  The input parameters are:\n\
  -x N              Use a mesh in the x direction of N.  \n\
  -c N              Use N V-cycles.  \n\
  -l N              Use N Levels.  \n\
  -smooths N        Use N pre smooths and N post smooths.  \n\
  -j                Use Jacobi smoother.  \n\
  -a use additive multigrid \n\
  -f use full multigrid (preconditioner variant) \n\
This example also demonstrates matrix-free methods\n\n";

/*
  This is not a good example to understand the use of multigrid with PETSc.
*/

#include <petscksp.h>

PetscErrorCode residual(Mat, Vec, Vec, Vec);
PetscErrorCode gauss_seidel(PC, Vec, Vec, Vec, PetscReal, PetscReal, PetscReal, PetscInt, PetscBool, PetscInt *, PCRichardsonConvergedReason *);
PetscErrorCode jacobi_smoother(PC, Vec, Vec, Vec, PetscReal, PetscReal, PetscReal, PetscInt, PetscBool, PetscInt *, PCRichardsonConvergedReason *);
PetscErrorCode interpolate(Mat, Vec, Vec, Vec);
PetscErrorCode restrct(Mat, Vec, Vec);
PetscErrorCode Create1dLaplacian(PetscInt, Mat *);
PetscErrorCode CalculateRhs(Vec);
PetscErrorCode CalculateError(Vec, Vec, Vec, PetscReal *);
PetscErrorCode CalculateSolution(PetscInt, Vec *);
PetscErrorCode amult(Mat, Vec, Vec);
PetscErrorCode apply_pc(PC, Vec, Vec);

int main(int Argc, char **Args)
{
  PetscInt    x_mesh = 15, levels = 3, cycles = 1, use_jacobi = 0;
  PetscInt    i, smooths = 1, *N, its;
  PCMGType    am = PC_MG_MULTIPLICATIVE;
  Mat         cmat, mat[20], fmat;
  KSP         cksp, ksp[20], kspmg;
  PetscReal   e[3]; /* l_2 error,max error, residual */
  const char *shellname;
  Vec         x, solution, X[20], R[20], B[20];
  PC          pcmg, pc;
  PetscBool   flg;

  PetscCall(PetscInitialize(&Argc, &Args, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-x", &x_mesh, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-l", &levels, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-c", &cycles, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-smooths", &smooths, NULL));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-a", &flg));

  if (flg) am = PC_MG_ADDITIVE;
  PetscCall(PetscOptionsHasName(NULL, NULL, "-f", &flg));
  if (flg) am = PC_MG_FULL;
  PetscCall(PetscOptionsHasName(NULL, NULL, "-j", &flg));
  if (flg) use_jacobi = 1;

  PetscCall(PetscMalloc1(levels, &N));
  N[0] = x_mesh;
  for (i = 1; i < levels; i++) {
    N[i] = N[i - 1] / 2;
    PetscCheck(N[i] >= 1, PETSC_COMM_WORLD, PETSC_ERR_USER, "Too many levels or N is not large enough");
  }

  PetscCall(Create1dLaplacian(N[levels - 1], &cmat));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &kspmg));
  PetscCall(KSPGetPC(kspmg, &pcmg));
  PetscCall(KSPSetFromOptions(kspmg));
  PetscCall(PCSetType(pcmg, PCMG));
  PetscCall(PCMGSetLevels(pcmg, levels, NULL));
  PetscCall(PCMGSetType(pcmg, am));

  PetscCall(PCMGGetCoarseSolve(pcmg, &cksp));
  PetscCall(KSPSetOperators(cksp, cmat, cmat));
  PetscCall(KSPGetPC(cksp, &pc));
  PetscCall(PCSetType(pc, PCLU));
  PetscCall(KSPSetType(cksp, KSPPREONLY));

  /* zero is finest level */
  for (i = 0; i < levels - 1; i++) {
    Mat dummy;

    PetscCall(PCMGSetResidual(pcmg, levels - 1 - i, residual, NULL));
    PetscCall(MatCreateShell(PETSC_COMM_WORLD, N[i + 1], N[i], N[i + 1], N[i], NULL, &mat[i]));
    PetscCall(MatShellSetOperation(mat[i], MATOP_MULT, (void (*)(void))restrct));
    PetscCall(MatShellSetOperation(mat[i], MATOP_MULT_TRANSPOSE_ADD, (void (*)(void))interpolate));
    PetscCall(PCMGSetInterpolation(pcmg, levels - 1 - i, mat[i]));
    PetscCall(PCMGSetRestriction(pcmg, levels - 1 - i, mat[i]));
    PetscCall(PCMGSetCycleTypeOnLevel(pcmg, levels - 1 - i, (PCMGCycleType)cycles));

    /* set smoother */
    PetscCall(PCMGGetSmoother(pcmg, levels - 1 - i, &ksp[i]));
    PetscCall(KSPGetPC(ksp[i], &pc));
    PetscCall(PCSetType(pc, PCSHELL));
    PetscCall(PCShellSetName(pc, "user_precond"));
    PetscCall(PCShellGetName(pc, &shellname));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "level=%" PetscInt_FMT ", PCShell name is %s\n", i, shellname));

    /* this is not used unless different options are passed to the solver */
    PetscCall(MatCreateShell(PETSC_COMM_WORLD, N[i], N[i], N[i], N[i], NULL, &dummy));
    PetscCall(MatShellSetOperation(dummy, MATOP_MULT, (void (*)(void))amult));
    PetscCall(KSPSetOperators(ksp[i], dummy, dummy));
    PetscCall(MatDestroy(&dummy));

    /*
        We override the matrix passed in by forcing it to use Richardson with
        a user provided application. This is non-standard and this practice
        should be avoided.
    */
    PetscCall(PCShellSetApply(pc, apply_pc));
    PetscCall(PCShellSetApplyRichardson(pc, gauss_seidel));
    if (use_jacobi) PetscCall(PCShellSetApplyRichardson(pc, jacobi_smoother));
    PetscCall(KSPSetType(ksp[i], KSPRICHARDSON));
    PetscCall(KSPSetInitialGuessNonzero(ksp[i], PETSC_TRUE));
    PetscCall(KSPSetTolerances(ksp[i], PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, smooths));

    PetscCall(VecCreateSeq(PETSC_COMM_SELF, N[i], &x));

    X[levels - 1 - i] = x;
    if (i > 0) PetscCall(PCMGSetX(pcmg, levels - 1 - i, x));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, N[i], &x));

    B[levels - 1 - i] = x;
    if (i > 0) PetscCall(PCMGSetRhs(pcmg, levels - 1 - i, x));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, N[i], &x));

    R[levels - 1 - i] = x;

    PetscCall(PCMGSetR(pcmg, levels - 1 - i, x));
  }
  /* create coarse level vectors */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, N[levels - 1], &x));
  PetscCall(PCMGSetX(pcmg, 0, x));
  X[0] = x;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, N[levels - 1], &x));
  PetscCall(PCMGSetRhs(pcmg, 0, x));
  B[0] = x;

  /* create matrix multiply for finest level */
  PetscCall(MatCreateShell(PETSC_COMM_WORLD, N[0], N[0], N[0], N[0], NULL, &fmat));
  PetscCall(MatShellSetOperation(fmat, MATOP_MULT, (void (*)(void))amult));
  PetscCall(KSPSetOperators(kspmg, fmat, fmat));

  PetscCall(CalculateSolution(N[0], &solution));
  PetscCall(CalculateRhs(B[levels - 1]));
  PetscCall(VecSet(X[levels - 1], 0.0));

  PetscCall(residual((Mat)0, B[levels - 1], X[levels - 1], R[levels - 1]));
  PetscCall(CalculateError(solution, X[levels - 1], R[levels - 1], e));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "l_2 error %g max error %g resi %g\n", (double)e[0], (double)e[1], (double)e[2]));

  PetscCall(KSPSolve(kspmg, B[levels - 1], X[levels - 1]));
  PetscCall(KSPGetIterationNumber(kspmg, &its));
  PetscCall(residual((Mat)0, B[levels - 1], X[levels - 1], R[levels - 1]));
  PetscCall(CalculateError(solution, X[levels - 1], R[levels - 1], e));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "its %" PetscInt_FMT " l_2 error %g max error %g resi %g\n", its, (double)e[0], (double)e[1], (double)e[2]));

  PetscCall(PetscFree(N));
  PetscCall(VecDestroy(&solution));

  /* note we have to keep a list of all vectors allocated, this is
     not ideal, but putting it in MGDestroy is not so good either*/
  for (i = 0; i < levels; i++) {
    PetscCall(VecDestroy(&X[i]));
    PetscCall(VecDestroy(&B[i]));
    if (i) PetscCall(VecDestroy(&R[i]));
  }
  for (i = 0; i < levels - 1; i++) PetscCall(MatDestroy(&mat[i]));
  PetscCall(MatDestroy(&cmat));
  PetscCall(MatDestroy(&fmat));
  PetscCall(KSPDestroy(&kspmg));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode residual(Mat mat, Vec bb, Vec xx, Vec rr)
{
  PetscInt           i, n1;
  PetscScalar       *x, *r;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetSize(bb, &n1));
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  PetscCall(VecGetArray(rr, &r));
  n1--;
  r[0]  = b[0] + x[1] - 2.0 * x[0];
  r[n1] = b[n1] + x[n1 - 1] - 2.0 * x[n1];
  for (i = 1; i < n1; i++) r[i] = b[i] + x[i + 1] + x[i - 1] - 2.0 * x[i];
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(VecRestoreArray(rr, &r));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode amult(Mat mat, Vec xx, Vec yy)
{
  PetscInt           i, n1;
  PetscScalar       *y;
  const PetscScalar *x;

  PetscFunctionBegin;
  PetscCall(VecGetSize(xx, &n1));
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(yy, &y));
  n1--;
  y[0]  = -x[1] + 2.0 * x[0];
  y[n1] = -x[n1 - 1] + 2.0 * x[n1];
  for (i = 1; i < n1; i++) y[i] = -x[i + 1] - x[i - 1] + 2.0 * x[i];
  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(yy, &y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode apply_pc(PC pc, Vec bb, Vec xx)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Not implemented");
}

PetscErrorCode gauss_seidel(PC pc, Vec bb, Vec xx, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt m, PetscBool guesszero, PetscInt *its, PCRichardsonConvergedReason *reason)
{
  PetscInt           i, n1;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  *its    = m;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  PetscCall(VecGetSize(bb, &n1));
  n1--;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  while (m--) {
    x[0] = .5 * (x[1] + b[0]);
    for (i = 1; i < n1; i++) x[i] = .5 * (x[i + 1] + x[i - 1] + b[i]);
    x[n1] = .5 * (x[n1 - 1] + b[n1]);
    for (i = n1 - 1; i > 0; i--) x[i] = .5 * (x[i + 1] + x[i - 1] + b[i]);
    x[0] = .5 * (x[1] + b[0]);
  }
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode jacobi_smoother(PC pc, Vec bb, Vec xx, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt m, PetscBool guesszero, PetscInt *its, PCRichardsonConvergedReason *reason)
{
  PetscInt           i, n, n1;
  PetscScalar       *r, *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  *its    = m;
  *reason = PCRICHARDSON_CONVERGED_ITS;
  PetscCall(VecGetSize(bb, &n));
  n1 = n - 1;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  PetscCall(VecGetArray(w, &r));

  while (m--) {
    r[0] = .5 * (x[1] + b[0]);
    for (i = 1; i < n1; i++) r[i] = .5 * (x[i + 1] + x[i - 1] + b[i]);
    r[n1] = .5 * (x[n1 - 1] + b[n1]);
    for (i = 0; i < n; i++) x[i] = (2.0 * r[i] + x[i]) / 3.0;
  }
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(VecRestoreArray(w, &r));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*
   We know for this application that yy  and zz are the same
*/

PetscErrorCode interpolate(Mat mat, Vec xx, Vec yy, Vec zz)
{
  PetscInt           i, n, N, i2;
  PetscScalar       *y;
  const PetscScalar *x;

  PetscFunctionBegin;
  PetscCall(VecGetSize(yy, &N));
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(yy, &y));
  n = N / 2;
  for (i = 0; i < n; i++) {
    i2 = 2 * i;
    y[i2] += .5 * x[i];
    y[i2 + 1] += x[i];
    y[i2 + 2] += .5 * x[i];
  }
  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(yy, &y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode restrct(Mat mat, Vec rr, Vec bb)
{
  PetscInt           i, n, N, i2;
  PetscScalar       *b;
  const PetscScalar *r;

  PetscFunctionBegin;
  PetscCall(VecGetSize(rr, &N));
  PetscCall(VecGetArrayRead(rr, &r));
  PetscCall(VecGetArray(bb, &b));
  n = N / 2;

  for (i = 0; i < n; i++) {
    i2   = 2 * i;
    b[i] = (r[i2] + 2.0 * r[i2 + 1] + r[i2 + 2]);
  }
  PetscCall(VecRestoreArrayRead(rr, &r));
  PetscCall(VecRestoreArray(bb, &b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Create1dLaplacian(PetscInt n, Mat *mat)
{
  PetscScalar mone = -1.0, two = 2.0;
  PetscInt    i, idx;

  PetscFunctionBegin;
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, n, n, 3, NULL, mat));

  idx = n - 1;
  PetscCall(MatSetValues(*mat, 1, &idx, 1, &idx, &two, INSERT_VALUES));
  for (i = 0; i < n - 1; i++) {
    PetscCall(MatSetValues(*mat, 1, &i, 1, &i, &two, INSERT_VALUES));
    idx = i + 1;
    PetscCall(MatSetValues(*mat, 1, &idx, 1, &i, &mone, INSERT_VALUES));
    PetscCall(MatSetValues(*mat, 1, &i, 1, &idx, &mone, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CalculateRhs(Vec u)
{
  PetscInt    i, n;
  PetscReal   h;
  PetscScalar uu;

  PetscFunctionBegin;
  PetscCall(VecGetSize(u, &n));
  h = 1.0 / ((PetscReal)(n + 1));
  for (i = 0; i < n; i++) {
    uu = 2.0 * h * h;
    PetscCall(VecSetValues(u, 1, &i, &uu, INSERT_VALUES));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CalculateSolution(PetscInt n, Vec *solution)
{
  PetscInt    i;
  PetscReal   h, x = 0.0;
  PetscScalar uu;

  PetscFunctionBegin;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, solution));
  h = 1.0 / ((PetscReal)(n + 1));
  for (i = 0; i < n; i++) {
    x += h;
    uu = x * (1. - x);
    PetscCall(VecSetValues(*solution, 1, &i, &uu, INSERT_VALUES));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CalculateError(Vec solution, Vec u, Vec r, PetscReal *e)
{
  PetscFunctionBegin;
  PetscCall(VecNorm(r, NORM_2, e + 2));
  PetscCall(VecWAXPY(r, -1.0, u, solution));
  PetscCall(VecNorm(r, NORM_2, e));
  PetscCall(VecNorm(r, NORM_1, e + 1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   test:

TEST*/
