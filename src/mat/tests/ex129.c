/*
  Laplacian in 3D. Use for testing MatSolve routines.
  Modeled by the partial differential equation

   - Laplacian u = 1,0 < x,y,z < 1,

   with boundary conditions
   u = 1 for x = 0, x = 1, y = 0, y = 1, z = 0, z = 1.
*/

static char help[] = "This example is for testing different MatSolve routines :MatSolve(), MatSolveAdd(), MatSolveTranspose(), MatSolveTransposeAdd(), MatMatSolve(), and MatMatSolveTranspose(), including how they flag a solution they could not compute.\n\
Example usage: ./ex129 -mat_type aij -dof 2\n\n";

#include <petscdm.h>
#include <petscdmda.h>

extern PetscErrorCode ComputeMatrix(DM, Mat);
extern PetscErrorCode ComputeRHS(DM, Vec);
extern PetscErrorCode ComputeRHSMatrix(PetscInt, PetscInt, Mat *);

/*
   Every entry of a solution that could not be computed must have been flagged with positive infinity, see MatFlag() and VecFlag()
*/
static PetscErrorCode CheckInf(Mat X, const char name[])
{
  const PetscScalar *x;
  PetscInt           i, j, m, N, lda;

  PetscFunctionBeginUser;
  PetscCall(MatGetLocalSize(X, &m, NULL));
  PetscCall(MatGetSize(X, NULL, &N));
  PetscCall(MatDenseGetLDA(X, &lda));
  PetscCall(MatDenseGetArrayRead(X, &x));
  for (j = 0; j < N; j++)
    for (i = 0; i < m; i++)
      PetscCheck(PetscIsInfReal(PetscRealPart(x[i + j * lda])) && PetscRealPart(x[i + j * lda]) > 0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s left entry (%" PetscInt_FMT ",%" PetscInt_FMT ") unflagged after a failed factorization", name, i, j);
  PetscCall(MatDenseRestoreArrayRead(X, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   A factorization that hit a zero pivot must make MatMatSolve() and MatMatSolveTranspose() flag every column of X, exactly as MatSolve() flags x
*/
static PetscErrorCode TestZeroPivot(PetscInt n, PetscInt nrhs)
{
  Mat                A, F, B, X;
  Vec                b, x;
  IS                 perm, iperm;
  MatFactorInfo      info;
  const PetscScalar *xx;
  PetscInt           i;

  PetscFunctionBeginUser;
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, n, n, 1, NULL, &A));
  /* an explicit zero on the diagonal gives a numeric, rather than structural, zero pivot */
  for (i = 0; i < n; i++) PetscCall(MatSetValue(A, i, i, i == n / 2 ? 0.0 : 1.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatFactorInfoInitialize(&info));
  PetscCall(MatGetOrdering(A, MATORDERINGNATURAL, &perm, &iperm));
  PetscCall(MatGetFactor(A, MATSOLVERPETSC, MAT_FACTOR_LU, &F));
  PetscCall(MatLUFactorSymbolic(F, A, perm, iperm, &info));
  /* the zero pivot is divided by during the factorization, so do not trap floating point exceptions until the solves are done */
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCall(MatLUFactorNumeric(F, A, &info));

  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecSet(b, 1.0));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, n, nrhs, NULL, &B));
  PetscCall(MatZeroEntries(B));
  PetscCall(MatDuplicate(B, MAT_COPY_VALUES, &X)); /* start from zeros so that an untouched X cannot pass the checks below */

  /* the reference behaviour that the block solves must reproduce */
  PetscCall(MatSolve(F, b, x));
  PetscCall(VecGetArrayRead(x, &xx));
  for (i = 0; i < n; i++) PetscCheck(PetscIsInfReal(PetscRealPart(xx[i])) && PetscRealPart(xx[i]) > 0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatSolve() left entry %" PetscInt_FMT " unflagged after a failed factorization", i);
  PetscCall(VecRestoreArrayRead(x, &xx));

  PetscCall(MatMatSolve(F, B, X));
  PetscCall(CheckInf(X, "MatMatSolve()"));

  PetscCall(MatZeroEntries(X));
  PetscCall(MatMatSolveTranspose(F, B, X));
  PetscCall(CheckInf(X, "MatMatSolveTranspose()"));
  PetscCall(PetscFPTrapPop());

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&F));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&X));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(ISDestroy(&perm));
  PetscCall(ISDestroy(&iperm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  PetscMPIInt   size;
  Vec           x, b, y, b1;
  DM            da;
  Mat           A, F, RHS, X, C1;
  MatFactorInfo info;
  IS            perm, iperm;
  PetscInt      dof = 1, M = 8, m, n, nrhs;
  PetscScalar   one = 1.0;
  PetscReal     norm, tol = 1000 * PETSC_MACHINE_EPSILON;
  PetscBool     InplaceLU = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only");
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dof", &dof, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-M", &M, NULL));

  PetscCall(TestZeroPivot(4, 2));

  PetscCall(DMDACreate(PETSC_COMM_WORLD, &da));
  PetscCall(DMSetDimension(da, 3));
  PetscCall(DMDASetBoundaryType(da, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE));
  PetscCall(DMDASetStencilType(da, DMDA_STENCIL_STAR));
  PetscCall(DMDASetSizes(da, M, M, M));
  PetscCall(DMDASetNumProcs(da, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(DMDASetDof(da, dof));
  PetscCall(DMDASetStencilWidth(da, 1));
  PetscCall(DMDASetOwnershipRanges(da, NULL, NULL, NULL));
  PetscCall(DMSetMatType(da, MATBAIJ));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  PetscCall(DMCreateGlobalVector(da, &x));
  PetscCall(DMCreateGlobalVector(da, &b));
  PetscCall(VecDuplicate(b, &y));
  PetscCall(ComputeRHS(da, b));
  PetscCall(VecSet(y, one));
  PetscCall(DMCreateMatrix(da, &A));
  PetscCall(ComputeMatrix(da, A));
  PetscCall(MatGetSize(A, &m, &n));
  nrhs = 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nrhs", &nrhs, NULL));
  PetscCall(ComputeRHSMatrix(m, nrhs, &RHS));
  PetscCall(MatDuplicate(RHS, MAT_DO_NOT_COPY_VALUES, &X));

  PetscCall(MatGetOrdering(A, MATORDERINGND, &perm, &iperm));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-inplacelu", &InplaceLU, NULL));
  PetscCall(MatFactorInfoInitialize(&info));
  if (!InplaceLU) {
    PetscCall(MatGetFactor(A, MATSOLVERPETSC, MAT_FACTOR_LU, &F));
    info.fill = 5.0;
    PetscCall(MatLUFactorSymbolic(F, A, perm, iperm, &info));
    PetscCall(MatLUFactorNumeric(F, A, &info));
  } else { /* Test inplace factorization */
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &F));
    PetscCall(MatLUFactor(F, perm, iperm, &info));
  }

  PetscCall(VecDuplicate(y, &b1));

  /* MatSolve */
  PetscCall(MatSolve(F, b, x));
  PetscCall(MatMult(A, x, b1));
  PetscCall(VecAXPY(b1, -1.0, b));
  PetscCall(VecNorm(b1, NORM_2, &norm));
  if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatSolve              : Error of norm %g\n", (double)norm));

  /* MatSolveTranspose */
  PetscCall(MatSolveTranspose(F, b, x));
  PetscCall(MatMultTranspose(A, x, b1));
  PetscCall(VecAXPY(b1, -1.0, b));
  PetscCall(VecNorm(b1, NORM_2, &norm));
  if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatSolveTranspose     : Error of norm %g\n", (double)norm));

  /* MatSolveAdd */
  PetscCall(MatSolveAdd(F, b, y, x));
  PetscCall(MatMult(A, y, b1));
  PetscCall(VecScale(b1, -1.0));
  PetscCall(MatMultAdd(A, x, b1, b1));
  PetscCall(VecAXPY(b1, -1.0, b));
  PetscCall(VecNorm(b1, NORM_2, &norm));
  if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatSolveAdd           : Error of norm %g\n", (double)norm));

  /* MatSolveTransposeAdd */
  PetscCall(MatSolveTransposeAdd(F, b, y, x));
  PetscCall(MatMultTranspose(A, y, b1));
  PetscCall(VecScale(b1, -1.0));
  PetscCall(MatMultTransposeAdd(A, x, b1, b1));
  PetscCall(VecAXPY(b1, -1.0, b));
  PetscCall(VecNorm(b1, NORM_2, &norm));
  if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatSolveTransposeAdd  : Error of norm %g\n", (double)norm));

  /* MatMatSolve */
  PetscCall(MatMatSolve(F, RHS, X));
  PetscCall(MatMatMult(A, X, MAT_INITIAL_MATRIX, 2.0, &C1));
  PetscCall(MatAXPY(C1, -1.0, RHS, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(C1, NORM_FROBENIUS, &norm));
  if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatMatSolve           : Error of norm %g\n", (double)norm));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&b1));
  PetscCall(VecDestroy(&y));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&F));
  PetscCall(MatDestroy(&RHS));
  PetscCall(MatDestroy(&C1));
  PetscCall(MatDestroy(&X));
  PetscCall(ISDestroy(&perm));
  PetscCall(ISDestroy(&iperm));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode ComputeRHS(DM da, Vec b)
{
  PetscInt    mx, my, mz;
  PetscScalar h;

  PetscFunctionBegin;
  PetscCall(DMDAGetInfo(da, NULL, &mx, &my, &mz, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  h = 1.0 / ((mx - 1) * (my - 1) * (mz - 1));
  PetscCall(VecSet(b, h));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeRHSMatrix(PetscInt m, PetscInt nrhs, Mat *C)
{
  PetscRandom  rand;
  Mat          RHS;
  PetscScalar *array, rval;
  PetscInt     i, k;

  PetscFunctionBegin;
  PetscCall(MatCreate(PETSC_COMM_WORLD, &RHS));
  PetscCall(MatSetSizes(RHS, m, PETSC_DECIDE, PETSC_DECIDE, nrhs));
  PetscCall(MatSetType(RHS, MATSEQDENSE));
  PetscCall(MatSetUp(RHS));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(MatDenseGetArray(RHS, &array));
  for (i = 0; i < m; i++) {
    PetscCall(PetscRandomGetValue(rand, &rval));
    array[i] = rval;
  }
  if (nrhs > 1) {
    for (k = 1; k < nrhs; k++) {
      for (i = 0; i < m; i++) array[m * k + i] = array[i];
    }
  }
  PetscCall(MatDenseRestoreArray(RHS, &array));
  PetscCall(MatAssemblyBegin(RHS, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(RHS, MAT_FINAL_ASSEMBLY));
  *C = RHS;
  PetscCall(PetscRandomDestroy(&rand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeMatrix(DM da, Mat B)
{
  PetscInt     i, j, k, mx, my, mz, xm, ym, zm, xs, ys, zs, dof, k1, k2, k3;
  PetscScalar *v, *v_neighbor, Hx, Hy, Hz, HxHydHz, HyHzdHx, HxHzdHy, r1, r2;
  MatStencil   row, col;
  PetscRandom  rand;

  PetscFunctionBegin;
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetSeed(rand, 1));
  PetscCall(PetscRandomSetInterval(rand, -.001, .001));
  PetscCall(PetscRandomSetFromOptions(rand));

  PetscCall(DMDAGetInfo(da, NULL, &mx, &my, &mz, NULL, NULL, NULL, &dof, NULL, NULL, NULL, NULL, NULL));
  /* For simplicity, this example only works on mx=my=mz */
  PetscCheck(mx == my && mx == mz, PETSC_COMM_SELF, PETSC_ERR_SUP, "This example only works with mx %" PetscInt_FMT " = my %" PetscInt_FMT " = mz %" PetscInt_FMT, mx, my, mz);

  Hx      = 1.0 / (PetscReal)(mx - 1);
  Hy      = 1.0 / (PetscReal)(my - 1);
  Hz      = 1.0 / (PetscReal)(mz - 1);
  HxHydHz = Hx * Hy / Hz;
  HxHzdHy = Hx * Hz / Hy;
  HyHzdHx = Hy * Hz / Hx;

  PetscCall(PetscMalloc1(2 * dof * dof + 1, &v));
  v_neighbor = v + dof * dof;
  PetscCall(PetscArrayzero(v, 2 * dof * dof + 1));
  k3 = 0;
  for (k1 = 0; k1 < dof; k1++) {
    for (k2 = 0; k2 < dof; k2++) {
      if (k1 == k2) {
        v[k3]          = 2.0 * (HxHydHz + HxHzdHy + HyHzdHx);
        v_neighbor[k3] = -HxHydHz;
      } else {
        PetscCall(PetscRandomGetValue(rand, &r1));
        PetscCall(PetscRandomGetValue(rand, &r2));

        v[k3]          = r1;
        v_neighbor[k3] = r2;
      }
      k3++;
    }
  }
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));

  for (k = zs; k < zs + zm; k++) {
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        row.i = i;
        row.j = j;
        row.k = k;
        if (i == 0 || j == 0 || k == 0 || i == mx - 1 || j == my - 1 || k == mz - 1) { /* boundary points */
          PetscCall(MatSetValuesBlockedStencil(B, 1, &row, 1, &row, v, INSERT_VALUES));
        } else { /* interior points */
          /* center */
          col.i = i;
          col.j = j;
          col.k = k;
          PetscCall(MatSetValuesBlockedStencil(B, 1, &row, 1, &col, v, INSERT_VALUES));

          /* x neighbors */
          col.i = i - 1;
          col.j = j;
          col.k = k;
          PetscCall(MatSetValuesBlockedStencil(B, 1, &row, 1, &col, v_neighbor, INSERT_VALUES));
          col.i = i + 1;
          col.j = j;
          col.k = k;
          PetscCall(MatSetValuesBlockedStencil(B, 1, &row, 1, &col, v_neighbor, INSERT_VALUES));

          /* y neighbors */
          col.i = i;
          col.j = j - 1;
          col.k = k;
          PetscCall(MatSetValuesBlockedStencil(B, 1, &row, 1, &col, v_neighbor, INSERT_VALUES));
          col.i = i;
          col.j = j + 1;
          col.k = k;
          PetscCall(MatSetValuesBlockedStencil(B, 1, &row, 1, &col, v_neighbor, INSERT_VALUES));

          /* z neighbors */
          col.i = i;
          col.j = j;
          col.k = k - 1;
          PetscCall(MatSetValuesBlockedStencil(B, 1, &row, 1, &col, v_neighbor, INSERT_VALUES));
          col.i = i;
          col.j = j;
          col.k = k + 1;
          PetscCall(MatSetValuesBlockedStencil(B, 1, &row, 1, &col, v_neighbor, INSERT_VALUES));
        }
      }
    }
  }
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscFree(v));
  PetscCall(PetscRandomDestroy(&rand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   test:
      args: -dm_mat_type aij -dof 1
      output_file: output/empty.out

   test:
      suffix: 2
      args: -dm_mat_type aij -dof 1 -inplacelu
      output_file: output/empty.out

TEST*/
