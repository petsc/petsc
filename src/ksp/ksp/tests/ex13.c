/* These tests taken from https://stanford.edu/group/SOL/software/minresqlp/minresqlp-matlab/CPS11.zip
   Examples in CSP11/Algorithms/MINRESQLP/minresQLP.m comments */
static char help[] = "Tests MINRES-QLP.\n\n";

#include <petscksp.h>

static PetscErrorCode Get2DStencil(PetscInt i, PetscInt j, PetscInt n, PetscInt idxs[])
{
  PetscInt k = 0;

  PetscFunctionBeginUser;
  for (k = 0; k < 9; k++) idxs[k] = -1;
  k = 0;
  for (PetscInt k1 = -1; k1 <= 1; k1++)
    for (PetscInt k2 = -1; k2 <= 1; k2++)
      if (i + k1 >= 0 && i + k1 < n && j + k2 >= 0 && j + k2 < n) idxs[k++] = n * (i + k1) + (j + k2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Vec         x, b;
  Mat         A, P;
  KSP         ksp;
  PC          pc;
  PetscInt    testcase = 0, m, nnz, pnnz;
  PetscMPIInt rank;
  PCType      pctype;
  PetscBool   flg;
  PetscReal   radius = 0.0;
  PetscReal   rtol   = PETSC_DEFAULT;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-testcase", &testcase, NULL));
  switch (testcase) {
  case 1:
    m      = 100;
    nnz    = 3;
    pnnz   = 1;
    pctype = PCMAT;
    rtol   = 1.e-10;
    break;
  case 2:
    m      = 2500;
    nnz    = 9;
    pnnz   = 0;
    pctype = PCNONE;
    rtol   = 1.e-5;
    radius = 1.e2;
    break;
  case 3:
    m      = 21;
    nnz    = 1;
    pnnz   = 0;
    pctype = PCNONE;
    radius = 1.e2;
    break;
  default: /* Example 7.1 in https://stanford.edu/group/SOL/software/minresqlp/MINRESQLP-SISC-2011.pdf */
    m      = 4;
    nnz    = 3;
    pnnz   = 1;
    pctype = PCMAT;
    break;
  }

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, m));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatMPIAIJSetPreallocation(A, nnz, NULL, nnz, NULL));
  PetscCall(MatSeqAIJSetPreallocation(A, nnz, NULL));
  PetscCall(MatSetUp(A));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &P));
  PetscCall(MatSetSizes(P, PETSC_DECIDE, PETSC_DECIDE, m, m));
  PetscCall(MatSetFromOptions(P));
  PetscCall(MatMPIAIJSetPreallocation(P, pnnz, NULL, pnnz, NULL));
  PetscCall(MatSeqAIJSetPreallocation(P, pnnz, NULL));
  PetscCall(MatSetUp(P));

  PetscCall(MatCreateVecs(A, &x, &b));

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  /* dummy assemble */
  if (rank == 0) {
    PetscScalar *vals;
    PetscInt    *cols;
    PetscInt     row;
    PetscCall(PetscMalloc2(nnz, &cols, nnz, &vals));
    switch (testcase) {
    case 1:
      vals[0] = -2.0;
      vals[1] = 4.0;
      vals[2] = -2.0;
      for (row = 0; row < m; row++) {
        cols[0] = row - 1;
        cols[1] = row;
        cols[2] = row == m - 1 ? -1 : row + 1;
        PetscCall(MatSetValues(A, 1, &row, 3, cols, vals, INSERT_VALUES));
        PetscCall(MatSetValue(P, row, row, 1.0 / 4.0, INSERT_VALUES));
      }
      break;
    case 2:
      for (PetscInt i = 0; i < 9; i++) vals[i] = 1.0;
      for (PetscInt i = 0; i < 50; i++) {
        for (PetscInt j = 0; j < 50; j++) {
          PetscInt row = i * 50 + j;
          PetscCall(Get2DStencil(i, j, 50, cols));
          PetscCall(MatSetValues(A, 1, &row, 9, cols, vals, INSERT_VALUES));
        }
      }
      break;
    case 3:
      for (row = 0; row < m; row++) {
        if (row == 10) continue;
        vals[0] = row - 10.0;
        PetscCall(MatSetValue(A, row, row, vals[0], INSERT_VALUES));
      }
      break;
    default:
      vals[0] = vals[1] = vals[2] = 1.0;
      row                         = 0;
      cols[0]                     = 0;
      cols[1]                     = 1;
      PetscCall(MatSetValues(A, 1, &row, 2, cols, vals, INSERT_VALUES));
      PetscCall(VecSetValue(b, row, 6.0, INSERT_VALUES));
      PetscCall(MatSetValue(P, row, row, PetscSqr(0.84201), INSERT_VALUES));
      row     = 1;
      cols[0] = 0;
      cols[1] = 1;
      cols[2] = 2;
      PetscCall(MatSetValues(A, 1, &row, 3, cols, vals, INSERT_VALUES));
      PetscCall(VecSetValue(b, row, 9.0, INSERT_VALUES));
      PetscCall(MatSetValue(P, row, row, PetscSqr(0.81228), INSERT_VALUES));
      row     = 2;
      cols[0] = 1;
      cols[1] = 3;
      PetscCall(MatSetValues(A, 1, &row, 2, cols, vals, INSERT_VALUES));
      PetscCall(VecSetValue(b, row, 6.0, INSERT_VALUES));
      PetscCall(MatSetValue(P, row, row, PetscSqr(0.30957), INSERT_VALUES));
      row     = 3;
      cols[0] = 2;
      PetscCall(MatSetValues(A, 1, &row, 1, cols, vals, INSERT_VALUES));
      PetscCall(VecSetValue(b, row, 3.0, INSERT_VALUES));
      PetscCall(MatSetValue(P, row, row, PetscSqr(3.2303), INSERT_VALUES));
      break;
    }
    PetscCall(PetscFree2(cols, vals));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));

  /* rhs */
  switch (testcase) {
  case 1:
  case 2:
    PetscCall(VecSet(x, 1.0));
    PetscCall(MatMult(A, x, b));
    break;
  case 3:
    PetscCall(VecSet(b, 1.0));
    break;
  default:
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));
    break;
  }

  /* Create linear solver context */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, P));
  PetscCall(KSPSetTolerances(ksp, rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetType(ksp, KSPMINRES));
  PetscCall(KSPMINRESSetUseQLP(ksp, PETSC_TRUE));
  if (radius > 0.0) PetscCall(KSPMINRESSetRadius(ksp, radius));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, pctype));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));

  /* print info */
  PetscCall(PetscObjectTypeCompare((PetscObject)ksp, KSPMINRES, &flg));
  if (flg) {
    PetscCall(KSPMINRESGetUseQLP(ksp, &flg));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Using KSPMINRES%s\n", flg ? "-QLP" : ""));
  } else {
    KSPType ksptype;
    PetscCall(KSPGetType(ksp, &ksptype));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Using %s\n", ksptype));
  }

  /* solve */
  PetscCall(KSPSolve(ksp, b, x));

  /* test reuse */
  PetscCall(KSPSolve(ksp, b, x));

  /* Free work space. */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&P));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: qlp_sisc
      args: -ksp_converged_reason -ksp_minres_monitor -ksp_view_solution

   test:
      suffix: qlp_sisc_none
      args: -ksp_converged_reason -ksp_minres_monitor -ksp_view_solution -pc_type none

   test:
      suffix: qlp_1
      args: -ksp_converged_reason -testcase 1 -ksp_minres_monitor
      filter: sed -e "s/49    3/49    1/g" -e "s/50    3/50    1/g" -e "s/CONVERGED_HAPPY_BREAKDOWN/CONVERGED_RTOL/g"

   test:
      suffix: qlp_2
      args: -ksp_converged_reason -testcase 2 -ksp_minres_monitor

   test:
      suffix: qlp_3
      args: -ksp_converged_reason -testcase 3 -ksp_minres_monitor
      filter: sed -e "s/24    2/24    6/g" -e "s/50    3/50    1/g" -e "s/CONVERGED_RTOL/CONVERGED_STEP_LENGTH/g"

TEST*/
