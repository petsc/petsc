const char help[] = "Test correctness of MatLMVM implementations";

#include <petscksp.h>

static PetscErrorCode MatSolveHermitianTranspose(Mat B, Vec x, Vec y)
{
  Vec x_conj = x;

  PetscFunctionBegin;
  if (PetscDefined(USE_COMPLEX)) {
    PetscCall(VecDuplicate(x, &x_conj));
    PetscCall(VecCopy(x, x_conj));
    PetscCall(VecConjugate(x_conj));
  }
  PetscCall(MatSolveTranspose(B, x_conj, y));
  if (PetscDefined(USE_COMPLEX)) {
    PetscCall(VecDestroy(&x_conj));
    PetscCall(VecConjugate(y));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HermitianTransposeTest(Mat B, PetscRandom rand, PetscBool inverse)
{
  Vec         x, f, Bx, Bhf;
  PetscScalar dot_a, dot_b;
  PetscReal   x_norm, Bhf_norm, Bx_norm, f_norm;
  PetscReal   err;
  PetscReal   scale;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(B, &x, &f));
  PetscCall(VecSetRandom(x, rand));
  PetscCall(VecSetRandom(f, rand));
  PetscCall(MatCreateVecs(B, &Bhf, &Bx));
  PetscCall((inverse ? MatSolve : MatMult)(B, x, Bx));
  PetscCall((inverse ? MatSolveHermitianTranspose : MatMultHermitianTranspose)(B, f, Bhf));
  PetscCall(VecNorm(x, NORM_2, &x_norm));
  PetscCall(VecNorm(Bhf, NORM_2, &Bhf_norm));
  PetscCall(VecNorm(Bx, NORM_2, &Bx_norm));
  PetscCall(VecNorm(f, NORM_2, &f_norm));
  PetscCall(VecDot(x, Bhf, &dot_a));
  PetscCall(VecDot(Bx, f, &dot_b));
  err   = PetscAbsScalar(dot_a - dot_b);
  scale = PetscMax(x_norm * Bhf_norm, Bx_norm * f_norm);
  PetscCall(PetscInfo((PetscObject)B, "Hermitian transpose error %g, relative error %g \n", (double)err, (double)(err / scale)));
  PetscCheck(err <= PETSC_SMALL * scale, PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Hermitian transpose error %g", (double)err);
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&f));
  PetscCall(VecDestroy(&Bx));
  PetscCall(VecDestroy(&Bhf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InverseTest(Mat B, PetscRandom rand)
{
  Vec       x, Bx, BinvBx;
  PetscReal x_norm, Bx_norm, err;
  PetscReal scale;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(B, &x, &Bx));
  PetscCall(VecDuplicate(x, &BinvBx));
  PetscCall(VecSetRandom(x, rand));
  PetscCall(MatMult(B, x, Bx));
  PetscCall(MatSolve(B, Bx, BinvBx));
  PetscCall(VecNorm(x, NORM_2, &x_norm));
  PetscCall(VecNorm(Bx, NORM_2, &Bx_norm));
  PetscCall(VecAXPY(BinvBx, -1.0, x));
  PetscCall(VecNorm(BinvBx, NORM_2, &err));
  scale = PetscMax(x_norm, Bx_norm);
  PetscCall(PetscInfo((PetscObject)B, "Inverse error %g, relative error %g\n", (double)err, (double)(err / scale)));
  PetscCheck(err <= 100.0 * PETSC_SMALL * scale, PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Inverse error %g", (double)err);
  PetscCall(VecDestroy(&BinvBx));
  PetscCall(VecDestroy(&Bx));
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IsHermitianTest(Mat B, PetscRandom rand, PetscBool inverse)
{
  Vec         x, y, Bx, By;
  PetscScalar dot_a, dot_b;
  PetscReal   x_norm, By_norm, Bx_norm, y_norm;
  PetscReal   err;
  PetscReal   scale;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(B, &x, &y));
  PetscCall(VecDuplicate(x, &By));
  PetscCall(VecDuplicate(y, &Bx));
  PetscCall(VecSetRandom(x, rand));
  PetscCall(VecSetRandom(y, rand));
  PetscCall((inverse ? MatSolve : MatMult)(B, x, Bx));
  PetscCall((inverse ? MatSolve : MatMult)(B, y, By));
  PetscCall(VecNorm(x, NORM_2, &x_norm));
  PetscCall(VecNorm(By, NORM_2, &By_norm));
  PetscCall(VecNorm(Bx, NORM_2, &Bx_norm));
  PetscCall(VecNorm(y, NORM_2, &y_norm));
  PetscCall(VecDot(x, By, &dot_a));
  PetscCall(VecDot(Bx, y, &dot_b));
  err   = PetscAbsScalar(dot_a - dot_b);
  scale = PetscMax(x_norm * By_norm, Bx_norm * y_norm);
  PetscCall(PetscInfo((PetscObject)B, "Is Hermitian error %g, relative error %g\n", (double)err, (double)(err / scale)));
  PetscCheck(err <= PETSC_SMALL * scale, PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Is Hermitian error %g", (double)err);
  PetscCall(VecDestroy(&By));
  PetscCall(VecDestroy(&Bx));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SecantTest(Mat B, Vec dx, Vec df, PetscBool is_hermitian, PetscBool test_inverse)
{
  Vec       B_x;
  PetscReal err, scale;

  PetscFunctionBegin;
  if (!test_inverse) {
    PetscCall(VecDuplicate(df, &B_x));
    PetscCall(MatMult(B, dx, B_x));
    PetscCall(VecAXPY(B_x, -1.0, df));
    PetscCall(VecNorm(B_x, NORM_2, &err));
    PetscCall(VecNorm(df, NORM_2, &scale));
    PetscCall(PetscInfo((PetscObject)B, "Secant error %g, relative error %g\n", (double)err, (double)(err / scale)));
    PetscCheck(err <= 10.0 * PETSC_SMALL * scale, PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Secant error %g", (double)err);

    if (is_hermitian) {
      PetscCall(MatMultHermitianTranspose(B, dx, B_x));
      PetscCall(VecAXPY(B_x, -1.0, df));
      PetscCall(VecNorm(B_x, NORM_2, &err));
      PetscCall(PetscInfo((PetscObject)B, "Adjoint secant error %g, relative error %g\n", (double)err, (double)(err / scale)));
      PetscCheck(err <= 10.0 * PETSC_SMALL * scale, PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Adjoint secant error %g", (double)err);
    }
  } else {
    PetscCall(VecDuplicate(df, &B_x));
    PetscCall(MatSolve(B, df, B_x));
    PetscCall(VecAXPY(B_x, -1.0, dx));

    PetscCall(VecNorm(B_x, NORM_2, &err));
    PetscCall(VecNorm(dx, NORM_2, &scale));
    PetscCall(PetscInfo((PetscObject)B, "Inverse secant error %g, relative error %g\n", (double)err, (double)(err / scale)));
    PetscCheck(err <= PETSC_SMALL * scale, PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Inverse secant error %g", (double)err);

    if (is_hermitian) {
      PetscCall(MatSolveHermitianTranspose(B, df, B_x));
      PetscCall(VecAXPY(B_x, -1.0, dx));
      PetscCall(VecNorm(B_x, NORM_2, &err));
      PetscCall(PetscInfo((PetscObject)B, "Adjoint inverse secant error %g, relative error %g\n", (double)err, (double)(err / scale)));
      PetscCheck(err <= PETSC_SMALL * scale, PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Adjoint inverse secant error %g", (double)err);
    }
  }
  PetscCall(VecDestroy(&B_x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RankOneAXPY(Mat C, PetscScalar alpha, Vec x, Vec y)
{
  PetscInt           m, n, M, N;
  Mat                col_mat, row_mat;
  const PetscScalar *x_a, *y_a;
  PetscScalar       *x_mat_a, *y_mat_a;
  Mat                outer_product;

  PetscFunctionBegin;
  PetscCall(MatGetSize(C, &M, &N));
  PetscCall(MatGetLocalSize(C, &m, &n));

  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)C), m, PETSC_DECIDE, M, 1, NULL, &col_mat));
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)C), n, PETSC_DECIDE, N, 1, NULL, &row_mat));

  PetscCall(VecGetArrayRead(x, &x_a));
  PetscCall(VecGetArrayRead(y, &y_a));

  PetscCall(MatDenseGetColumn(col_mat, 0, &x_mat_a));
  PetscCall(MatDenseGetColumn(row_mat, 0, &y_mat_a));

  PetscCall(PetscArraycpy(x_mat_a, x_a, m));
  PetscCall(PetscArraycpy(y_mat_a, y_a, n));

  PetscCall(MatDenseRestoreColumn(row_mat, &y_mat_a));
  PetscCall(MatDenseRestoreColumn(col_mat, &x_mat_a));

  PetscCall(VecRestoreArrayRead(y, &y_a));
  PetscCall(VecRestoreArrayRead(x, &x_a));

  PetscCall(MatConjugate(row_mat));
  PetscCall(MatMatTransposeMult(col_mat, row_mat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &outer_product));

  PetscCall(MatAXPY(C, alpha, outer_product, SAME_NONZERO_PATTERN));

  PetscCall(MatDestroy(&outer_product));
  PetscCall(MatDestroy(&row_mat));
  PetscCall(MatDestroy(&col_mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BroydenUpdate_Explicit(Mat B, PetscReal unused_, Vec s, Vec y)
{
  PetscScalar sts;
  Vec         ymBs;

  PetscFunctionBegin;
  PetscCall(VecDot(s, s, &sts));
  PetscCall(VecDuplicate(y, &ymBs));
  PetscCall(MatMult(B, s, ymBs));
  PetscCall(VecAYPX(ymBs, -1.0, y));
  PetscCall(RankOneAXPY(B, 1.0 / sts, ymBs, s));
  PetscCall(VecDestroy(&ymBs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BadBroydenUpdate_Explicit(Mat B, PetscReal unused_, Vec s, Vec y)
{
  PetscScalar ytBs;
  Vec         Bty, ymBs;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(y, &ymBs));
  PetscCall(VecDuplicate(s, &Bty));
  PetscCall(MatMult(B, s, ymBs));
  PetscCall(VecDot(ymBs, y, &ytBs));
  PetscCall(VecAYPX(ymBs, -1.0, y));
  PetscCall(MatMultHermitianTranspose(B, y, Bty));
  PetscCall(RankOneAXPY(B, 1.0 / ytBs, ymBs, Bty));
  PetscCall(VecDestroy(&Bty));
  PetscCall(VecDestroy(&ymBs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SymmetricBroydenUpdate_Explicit(Mat B, PetscReal phi, Vec s, Vec y)
{
  Vec         Bs;
  PetscScalar stBs, yts;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(y, &Bs));
  PetscCall(MatMult(B, s, Bs));
  PetscCall(VecDot(s, Bs, &stBs));
  PetscCall(VecDot(s, y, &yts));
  PetscCall(RankOneAXPY(B, (yts + phi * stBs) / (yts * yts), y, y));
  PetscCall(RankOneAXPY(B, -phi / yts, y, Bs));
  PetscCall(RankOneAXPY(B, -phi / yts, Bs, y));
  PetscCall(RankOneAXPY(B, (phi - 1.0) / stBs, Bs, Bs));
  PetscCall(VecDestroy(&Bs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BFGSUpdate_Explicit(Mat B, PetscReal unused_, Vec s, Vec y)
{
  PetscFunctionBegin;
  PetscCall(SymmetricBroydenUpdate_Explicit(B, 0.0, s, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DFPUpdate_Explicit(Mat B, PetscReal unused_, Vec s, Vec y)
{
  PetscFunctionBegin;
  PetscCall(SymmetricBroydenUpdate_Explicit(B, 1.0, s, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SR1Update_Explicit(Mat B, PetscReal unused_, Vec s, Vec y)
{
  PetscScalar ymBsts;
  Vec         Bty, ymBs;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(y, &ymBs));
  PetscCall(VecDuplicate(s, &Bty));
  PetscCall(MatMult(B, s, ymBs));
  PetscCall(VecAYPX(ymBs, -1.0, y));
  PetscCall(VecDot(s, ymBs, &ymBsts));
  PetscCall(RankOneAXPY(B, 1.0 / ymBsts, ymBs, ymBs));
  PetscCall(VecDestroy(&Bty));
  PetscCall(VecDestroy(&ymBs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_Solve(Mat A, Vec x, Vec y)
{
  Mat B;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, (void *)&B));
  PetscCall(MatSolve(B, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_J0Solve(Mat A, Vec x, Vec y)
{
  Mat B;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, (void *)&B));
  PetscCall(MatLMVMApplyJ0Inv(B, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatComputeInverseOperator(Mat B, Mat *B_k, PetscBool use_J0)
{
  Mat      Binv;
  PetscInt m, n, M, N;

  PetscFunctionBegin;
  PetscCall(MatGetSize(B, &M, &N));
  PetscCall(MatGetLocalSize(B, &m, &n));
  PetscCall(MatCreateShell(PetscObjectComm((PetscObject)B), m, n, M, N, (void *)B, &Binv));
  PetscCall(MatShellSetOperation(Binv, MATOP_MULT, (PetscErrorCodeFn *)(use_J0 ? MatMult_J0Solve : MatMult_Solve)));
  PetscCall(MatComputeOperator(Binv, MATDENSE, B_k));
  PetscCall(MatDestroy(&Binv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestUpdate(Mat B, PetscInt iter, PetscRandom rand, PetscBool is_hermitian, Vec dxs[], Vec dfs[], Mat B_0, Mat H_0, PetscErrorCode (*B_update)(Mat, PetscReal, Vec, Vec), PetscErrorCode (*H_update)(Mat, PetscReal, Vec, Vec), PetscReal phi)
{
  PetscLayout rmap, cmap;
  PetscBool   is_invertible;
  Mat         J_0;
  Vec         x, dx, f, x_prev, f_prev, df;
  PetscInt    m;
  PetscScalar rho;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetHistorySize(B, &m));
  PetscCall(MatGetLayouts(B, &rmap, &cmap));
  PetscCall(PetscLayoutCompare(rmap, cmap, &is_invertible));

  PetscCall(MatLMVMGetJ0(B, &J_0));

  dx = dxs[iter];
  df = dfs[iter];

  PetscCall(MatLMVMGetLastUpdate(B, &x_prev, &f_prev));
  PetscCall(VecDuplicate(x_prev, &x));
  PetscCall(VecDuplicate(f_prev, &f));
  PetscCall(VecSetRandom(dx, rand));
  PetscCall(VecSetRandom(df, rand));

  if (is_hermitian) {
    PetscCall(VecDot(dx, df, &rho));
    PetscCall(VecScale(dx, PetscAbsScalar(rho) / rho));
  } else {
    Vec Bdx;

    PetscCall(VecDuplicate(df, &Bdx));
    PetscCall(MatMult(B, dx, Bdx));
    PetscCall(VecDot(Bdx, df, &rho));
    PetscCall(VecScale(dx, PetscAbsScalar(rho) / rho));
    PetscCall(VecDestroy(&Bdx));
  }
  PetscCall(VecWAXPY(x, 1.0, x_prev, dx));
  PetscCall(VecWAXPY(f, 1.0, f_prev, df));
  PetscCall(MatLMVMUpdate(B, x, f));
  PetscCall(VecDestroy(&x));

  PetscCall(HermitianTransposeTest(B, rand, PETSC_FALSE));
  if (is_hermitian) PetscCall(IsHermitianTest(B, rand, PETSC_FALSE));
  if (is_invertible) {
    PetscCall(InverseTest(B, rand));
    PetscCall(HermitianTransposeTest(B, rand, PETSC_TRUE));
    if (is_hermitian) PetscCall(IsHermitianTest(B, rand, PETSC_TRUE));
  }

  if (iter < m) {
    PetscCall(SecantTest(B, dx, df, is_hermitian, PETSC_FALSE));
    if (is_invertible) PetscCall(SecantTest(B, dx, df, is_hermitian, PETSC_TRUE));
  }

  if (is_invertible) {
    // some implementations use internal caching to compute the product Hf: double check that this is working
    Vec       f_copy, Hf, Hf_copy;
    PetscReal norm, err;

    PetscCall(VecDuplicate(f, &f_copy));
    PetscCall(VecCopy(f, f_copy));
    PetscCall(VecDuplicate(x_prev, &Hf));
    PetscCall(VecDuplicate(x_prev, &Hf_copy));
    PetscCall(MatSolve(B, f, Hf));
    PetscCall(MatSolve(B, f_copy, Hf_copy));
    PetscCall(VecNorm(Hf_copy, NORM_2, &norm));
    PetscCall(VecAXPY(Hf, -1.0, Hf_copy));
    PetscCall(VecNorm(Hf, NORM_2, &err));
    PetscCheck(err <= PETSC_SMALL * norm, PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Gradient solve error %g", (double)err);

    PetscCall(VecDestroy(&Hf_copy));
    PetscCall(VecDestroy(&Hf));
    PetscCall(VecDestroy(&f_copy));
  }

  PetscCall(VecDestroy(&f));

  if (B_update) {
    PetscInt  oldest, next;
    Mat       B_k_exp;
    Mat       B_k;
    PetscReal norm, err;

    oldest = PetscMax(0, iter + 1 - m);
    next   = iter + 1;

    PetscCall(MatComputeOperator(B, MATDENSE, &B_k));
    PetscCall(MatDuplicate(B_0, MAT_COPY_VALUES, &B_k_exp));

    for (PetscInt i = oldest; i < next; i++) PetscCall((*B_update)(B_k_exp, phi, dxs[i], dfs[i]));
    PetscCall(MatNorm(B_k_exp, NORM_FROBENIUS, &norm));
    PetscCall(MatAXPY(B_k_exp, -1.0, B_k, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(B_k_exp, NORM_FROBENIUS, &err));
    PetscCall(PetscInfo((PetscObject)B_k, "Forward update error %g, relative error %g\n", (double)err, (double)(err / norm)));
    PetscCheck(err <= PETSC_SMALL * norm, PetscObjectComm((PetscObject)B_k), PETSC_ERR_PLIB, "Forward update error %g", (double)err);

    PetscCall(MatDestroy(&B_k_exp));
    PetscCall(MatDestroy(&B_k));
  }
  if (H_update) {
    PetscInt  oldest, next;
    Mat       H_k;
    Mat       H_k_exp;
    PetscReal norm, err;

    oldest = PetscMax(0, iter + 1 - m);
    next   = iter + 1;

    PetscCall(MatComputeInverseOperator(B, &H_k, PETSC_FALSE));
    PetscCall(MatDuplicate(H_0, MAT_COPY_VALUES, &H_k_exp));
    for (PetscInt i = oldest; i < next; i++) PetscCall((*H_update)(H_k_exp, phi, dfs[i], dxs[i]));
    PetscCall(MatNorm(H_k_exp, NORM_FROBENIUS, &norm));
    PetscCall(MatAXPY(H_k_exp, -1.0, H_k, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(H_k_exp, NORM_FROBENIUS, &err));
    PetscCall(PetscInfo((PetscObject)H_k, "Forward update error %g, relative error %g\n", (double)err, (double)(err / norm)));
    PetscCheck(err <= PETSC_SMALL * norm, PetscObjectComm((PetscObject)H_k), PETSC_ERR_PLIB, "Inverse update error %g", (double)err);

    PetscCall(MatDestroy(&H_k_exp));
    PetscCall(MatDestroy(&H_k));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetRandomWithShift(Mat J0, PetscRandom rand, PetscBool is_hermitian, PetscBool is_square)
{
  PetscFunctionBegin;
  PetscCall(MatSetRandom(J0, rand));
  if (is_hermitian) {
    Mat J0H;

    PetscCall(MatHermitianTranspose(J0, MAT_INITIAL_MATRIX, &J0H));
    PetscCall(MatAXPY(J0, 1.0, J0H, SAME_NONZERO_PATTERN));
    PetscCall(MatDestroy(&J0H));
  }
  if (is_square) {
    MPI_Comm    comm;
    PetscInt    N;
    PetscMPIInt count;
    Mat         J0copy;
    PetscReal  *real_eig, *imag_eig;
    KSP         kspeig;
    PC          pceig;
    PetscReal   shift;

    PetscCall(PetscObjectGetComm((PetscObject)J0, &comm));
    PetscCall(MatGetSize(J0, &N, NULL));
    PetscCall(MatDuplicate(J0, MAT_COPY_VALUES, &J0copy));
    PetscCall(PetscMalloc2(N, &real_eig, N, &imag_eig));
    PetscCall(KSPCreate(comm, &kspeig));
    if (is_hermitian) PetscCall(KSPSetType(kspeig, KSPMINRES));
    else PetscCall(KSPSetType(kspeig, KSPGMRES));
    PetscCall(KSPSetPCSide(kspeig, PC_LEFT));
    PetscCall(KSPGetPC(kspeig, &pceig));
    PetscCall(PCSetType(pceig, PCNONE));
    PetscCall(KSPSetOperators(kspeig, J0copy, J0copy));
    PetscCall(KSPComputeEigenvaluesExplicitly(kspeig, N, real_eig, imag_eig));
    PetscCall(PetscMPIIntCast(N, &count));
    PetscCallMPI(MPI_Bcast(real_eig, count, MPIU_REAL, 0, comm));
    PetscCall(PetscSortReal(N, real_eig));
    shift = PetscMax(2 * PetscAbsReal(real_eig[N - 1]), 2 * PetscAbsReal(real_eig[0]));
    PetscCall(MatShift(J0, shift));
    PetscCall(MatAssemblyBegin(J0, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J0, MAT_FINAL_ASSEMBLY));
    PetscCall(PetscFree2(real_eig, imag_eig));
    PetscCall(KSPDestroy(&kspeig));
    PetscCall(MatDestroy(&J0copy));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscInt    M = 15, N = 15, hist_size = 5, n_iter = 3 * hist_size, n_variable = n_iter / 2;
  PetscReal   phi = 0.0;
  MPI_Comm    comm;
  Mat         B, B_0, J_0, H_0 = NULL;
  PetscBool   B_is_h, B_is_h_known, is_hermitian, is_square, has_solve;
  PetscBool   is_brdn, is_badbrdn, is_dfp, is_bfgs, is_sr1, is_symbrdn, is_symbadbrdn, is_dbfgs, is_ddfp;
  PetscRandom rand;
  Vec         x, f;
  PetscLayout rmap, cmap;
  Vec        *dxs, *dfs;
  PetscErrorCode (*B_update)(Mat, PetscReal, Vec, Vec) = NULL;
  PetscErrorCode (*H_update)(Mat, PetscReal, Vec, Vec) = NULL;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscOptionsBegin(comm, NULL, help, NULL);
  PetscCall(PetscOptionsInt("-m", "# Matrix rows", NULL, M, &M, NULL));
  PetscCall(PetscOptionsInt("-n", "# Matrix columns", NULL, N, &N, NULL));
  PetscCall(PetscOptionsInt("-n_iter", "# test iterations", NULL, n_iter, &n_iter, NULL));
  PetscCall(PetscOptionsInt("-n_variable", "# test iterations where J0 changeschange J0 every iteration", NULL, n_variable, &n_variable, NULL));
  PetscOptionsEnd();

  PetscCall(PetscRandomCreate(comm, &rand));
  if (PetscDefined(USE_COMPLEX)) PetscCall(PetscRandomSetInterval(rand, -1.0 - PetscSqrtScalar(-1.0), 1.0 + PetscSqrtScalar(-1.0)));
  else PetscCall(PetscRandomSetInterval(rand, -1.0, 1.0));

  PetscCall(VecCreate(comm, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, N));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSetRandom(x, rand));
  PetscCall(VecCreate(comm, &f));
  PetscCall(VecSetFromOptions(f));
  PetscCall(VecSetSizes(f, PETSC_DECIDE, M));
  PetscCall(VecSetRandom(f, rand));

  PetscCall(MatCreate(comm, &B));
  PetscCall(MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, M, N));
  PetscCall(MatSetOptionsPrefix(B, "B_"));
  PetscCall(KSPInitializePackage());
  PetscCall(MatSetType(B, MATLMVMBROYDEN));
  PetscCall(MatLMVMSetHistorySize(B, hist_size));
  PetscCall(MatLMVMAllocate(B, x, f));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetUp(B));

  PetscCall(MatIsHermitianKnown(B, &B_is_h_known, &B_is_h));
  is_hermitian = (B_is_h_known && B_is_h) ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(MatGetLayouts(B, &rmap, &cmap));
  PetscCall(PetscLayoutCompare(rmap, cmap, &is_square));

  PetscCall(MatLMVMGetJ0(B, &J_0));
  PetscCall(MatSetRandomWithShift(J_0, rand, is_hermitian, is_square));

  PetscCall(PetscObjectTypeCompareAny((PetscObject)J_0, &has_solve, MATCONSTANTDIAGONAL, MATDIAGONAL, ""));
  if (is_square && !has_solve) {
    KSP ksp;
    PC  pc;

    PetscCall(MatLMVMGetJ0KSP(B, &ksp));
    if (is_hermitian) {
      PetscCall(KSPSetType(ksp, KSPCG));
      PetscCall(KSPCGSetType(ksp, KSP_CG_HERMITIAN));
    } else PetscCall(KSPSetType(ksp, KSPGMRES));
    PetscCall(KSPSetPCSide(ksp, PC_LEFT));
    PetscCall(KSPSetNormType(ksp, KSP_NORM_NONE));
    PetscCall(KSPSetTolerances(ksp, 0.0, 0.0, 0.0, N));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCNONE));
    PetscCall(MatLMVMSetJ0KSP(B, ksp));
  }

  PetscCall(MatViewFromOptions(B, NULL, "-view"));
  PetscCall(MatViewFromOptions(J_0, NULL, "-view"));

  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMBROYDEN, &is_brdn));
  if (is_brdn) {
    B_update = BroydenUpdate_Explicit;
    if (is_square) H_update = BadBroydenUpdate_Explicit;
  }

  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMBADBROYDEN, &is_badbrdn));
  if (is_badbrdn) {
    B_update = BadBroydenUpdate_Explicit;
    if (is_square) H_update = BroydenUpdate_Explicit;
  }

  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMDFP, &is_dfp));
  if (is_dfp) {
    B_update = DFPUpdate_Explicit;
    H_update = BFGSUpdate_Explicit;
  }

  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMBFGS, &is_bfgs));
  if (is_bfgs) {
    B_update = BFGSUpdate_Explicit;
    H_update = DFPUpdate_Explicit;
  }

  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMSYMBROYDEN, &is_symbrdn));
  if (is_symbrdn) {
    B_update = SymmetricBroydenUpdate_Explicit;
    PetscCall(MatLMVMSymBroydenGetPhi(B, &phi));
  }

  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMSYMBADBROYDEN, &is_symbadbrdn));
  if (is_symbadbrdn) {
    H_update = SymmetricBroydenUpdate_Explicit;
    PetscCall(MatLMVMSymBadBroydenGetPsi(B, &phi));
  }

  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMSR1, &is_sr1));
  if (is_sr1) {
    B_update = SR1Update_Explicit;
    H_update = SR1Update_Explicit;
  }

  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMDBFGS, &is_dbfgs));
  if (is_dbfgs) {
    B_update = BFGSUpdate_Explicit;
    H_update = DFPUpdate_Explicit;
  }

  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMDDFP, &is_ddfp));
  if (is_ddfp) {
    B_update = DFPUpdate_Explicit;
    H_update = BFGSUpdate_Explicit;
  }

  PetscCall(MatComputeOperator(J_0, MATDENSE, &B_0));
  if (is_square) PetscCall(MatComputeInverseOperator(B, &H_0, PETSC_TRUE));

  // Initialize with the first location
  PetscCall(MatLMVMUpdate(B, x, f));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&f));

  PetscCall(HermitianTransposeTest(B, rand, PETSC_FALSE));
  if (is_hermitian) PetscCall(IsHermitianTest(B, rand, PETSC_FALSE));
  if (is_square) {
    PetscCall(InverseTest(B, rand));
    if (is_hermitian) PetscCall(IsHermitianTest(B, rand, PETSC_TRUE));
    PetscCall(HermitianTransposeTest(B, rand, PETSC_TRUE));
  }

  PetscCall(PetscCalloc2(n_iter, &dxs, n_iter, &dfs));

  for (PetscInt i = 0; i < n_iter; i++) PetscCall(MatCreateVecs(B, &dxs[i], &dfs[i]));

  for (PetscInt i = 0; i < n_iter; i++) {
    PetscCall(TestUpdate(B, i, rand, is_hermitian, dxs, dfs, B_0, H_0, B_update, H_update, phi));
    if (i + n_variable >= n_iter) {
      PetscCall(MatSetRandomWithShift(J_0, rand, is_hermitian, is_square));
      PetscCall(MatLMVMSetJ0(B, J_0));
      PetscCall(MatDestroy(&B_0));
      PetscCall(MatDestroy(&H_0));
      PetscCall(MatComputeOperator(J_0, MATDENSE, &B_0));
      if (is_square) PetscCall(MatComputeInverseOperator(B, &H_0, PETSC_TRUE));
    }
  }

  for (PetscInt i = 0; i < n_iter; i++) {
    PetscCall(VecDestroy(&dxs[i]));
    PetscCall(VecDestroy(&dfs[i]));
  }

  PetscCall(PetscFree2(dxs, dfs));

  PetscCall(PetscRandomDestroy(&rand));

  PetscCall(MatDestroy(&H_0));
  PetscCall(MatDestroy(&B_0));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # TODO: enable hip complex if `#undef PETSC_HAVE_COMPLEX` can be removed from src/vec/is/sf/impls/basic/cupm/hip/sfcupm.hip.hpp

  # rectangular tests
  testset:
    requires: !single
    nsize: 2
    output_file: output/empty.out
    args: -m 15 -n 10 -B_mat_lmvm_J0_mat_type dense -B_mat_lmvm_mult_algorithm {{recursive dense compact_dense}}
    test:
      suffix: broyden_rectangular
      args: -B_mat_type lmvmbroyden
    test:
      suffix: badbroyden_rectangular
      args: -B_mat_type lmvmbadbroyden
    test:
      suffix: broyden_rectangular_cuda
      requires: cuda
      args: -B_mat_type lmvmbroyden -vec_type cuda -B_mat_lmvm_J0_mat_type densecuda
    test:
      suffix: badbroyden_rectangular_cuda
      requires: cuda
      args: -B_mat_type lmvmbadbroyden -vec_type cuda -B_mat_lmvm_J0_mat_type densecuda
    test:
      suffix: broyden_rectangular_hip
      requires: hip !complex
      args: -B_mat_type lmvmbroyden -vec_type hip -B_mat_lmvm_J0_mat_type densehip
    test:
      suffix: badbroyden_rectangular_hip
      requires: hip !complex
      args: -B_mat_type lmvmbadbroyden -vec_type hip -B_mat_lmvm_J0_mat_type densehip

  # square tests where compact_dense != dense
  testset:
    requires: !single
    nsize: 2
    output_file: output/empty.out
    args: -m 15 -n 15 -B_mat_lmvm_J0_mat_type {{constantdiagonal diagonal}} -B_mat_lmvm_mult_algorithm {{recursive dense compact_dense}} -B_mat_lmvm_cache_J0_products {{false true}}
    test:
      suffix: broyden_square
      args: -B_mat_type lmvmbroyden
    test:
      suffix: badbroyden_square
      args: -B_mat_type lmvmbadbroyden
    test:
      suffix: broyden_square_cuda
      requires: cuda
      args: -B_mat_type lmvmbroyden -vec_type cuda
    test:
      suffix: badbroyden_square_cuda
      requires: cuda
      args: -B_mat_type lmvmbadbroyden -vec_type cuda
    test:
      suffix: broyden_square_hip
      requires: hip !complex
      args: -B_mat_type lmvmbroyden -vec_type hip
    test:
      suffix: badbroyden_square_hip
      requires: hip !complex
      args: -B_mat_type lmvmbadbroyden -vec_type hip

  # square tests where compact_dense == dense
  testset:
    requires: !single
    nsize: 2
    output_file: output/empty.out
    args: -m 15 -n 15 -B_mat_lmvm_J0_mat_type dense -B_mat_lmvm_mult_algorithm {{recursive dense}}
    test:
      output_file: output/ex1_sr1.out
      suffix: sr1
      args: -B_mat_type lmvmsr1 -B_mat_lmvm_debug -B_view -B_mat_lmvm_cache_J0_products {{false true}}
      filter: grep -v "variant HERMITIAN"
    test:
      suffix: symbroyden
      args: -B_mat_lmvm_scale_type user -B_mat_type lmvmsymbroyden -B_mat_lmvm_phi {{0.0 0.6 1.0}}
    test:
      suffix: symbadbroyden
      args: -B_mat_lmvm_scale_type user -B_mat_type lmvmsymbadbroyden -B_mat_lmvm_psi {{0.0 0.6 1.0}}
    test:
      suffix: sr1_cuda
      requires: cuda
      args: -B_mat_type lmvmsr1 -vec_type cuda -B_mat_lmvm_J0_mat_type densecuda
    test:
      suffix: symbroyden_cuda
      requires: cuda
      args: -B_mat_lmvm_scale_type user -B_mat_type lmvmsymbroyden -B_mat_lmvm_phi {{0.0 0.6 1.0}} -vec_type cuda -B_mat_lmvm_J0_mat_type densecuda
    test:
      suffix: symbadbroyden_cuda
      requires: cuda
      args: -B_mat_lmvm_scale_type user -B_mat_type lmvmsymbadbroyden -B_mat_lmvm_psi {{0.0 0.6 1.0}} -vec_type cuda -B_mat_lmvm_J0_mat_type densecuda
    test:
      suffix: sr1_hip
      requires: hip !complex
      args: -B_mat_type lmvmsr1 -vec_type hip -B_mat_lmvm_J0_mat_type densehip
    test:
      suffix: symbroyden_hip
      requires: hip !complex
      args: -B_mat_lmvm_scale_type user -B_mat_type lmvmsymbroyden -B_mat_lmvm_phi {{0.0 0.6 1.0}} -vec_type hip -B_mat_lmvm_J0_mat_type densehip
    test:
      suffix: symbadbroyden_hip
      requires: hip !complex
      args: -B_mat_lmvm_scale_type user -B_mat_type lmvmsymbadbroyden -B_mat_lmvm_psi {{0.0 0.6 1.0}} -vec_type hip -B_mat_lmvm_J0_mat_type densehip

  testset:
    requires: !single
    nsize: 2
    output_file: output/empty.out
    args: -m 15 -n 15 -B_mat_lmvm_J0_mat_type {{constantdiagonal diagonal}} -B_mat_lmvm_mult_algorithm {{recursive dense compact_dense}} -B_mat_lmvm_scale_type user
    test:
      suffix: bfgs
      args: -B_mat_type lmvmbfgs
    test:
      suffix: dfp
      args: -B_mat_type lmvmdfp
    test:
      suffix: bfgs_cuda
      requires: cuda
      args: -B_mat_type lmvmbfgs -vec_type cuda
    test:
      suffix: dfp_cuda
      requires: cuda
      args: -B_mat_type lmvmdfp -vec_type cuda
    test:
      suffix: bfgs_hip
      requires: hip !complex
      args: -B_mat_type lmvmbfgs -vec_type hip
    test:
      suffix: dfp_hip
      requires: hip !complex
      args: -B_mat_type lmvmdfp -vec_type hip

  testset:
    requires: !single
    nsize: 2
    output_file: output/empty.out
    args: -m 15 -n 15 -B_mat_lmvm_J0_mat_type diagonal -B_mat_lmvm_scale_type user
    test:
      suffix: dbfgs
      args: -B_mat_type lmvmdbfgs -B_mat_lbfgs_recursive {{0 1}}
    test:
      suffix: ddfp
      args: -B_mat_type lmvmddfp -B_mat_ldfp_recursive {{0 1}}
    test:
      requires: cuda
      suffix: dbfgs_cuda
      args: -B_mat_type lmvmdbfgs -B_mat_lbfgs_recursive {{0 1}} -vec_type cuda
    test:
      requires: cuda
      suffix: ddfp_cuda
      args: -B_mat_type lmvmddfp -B_mat_ldfp_recursive {{0 1}} -vec_type cuda
    test:
      requires: hip !complex
      suffix: dbfgs_hip
      args: -B_mat_type lmvmdbfgs -B_mat_lbfgs_recursive {{0 1}} -vec_type hip
    test:
      requires: hip !complex
      suffix: ddfp_hip
      args: -B_mat_type lmvmddfp -B_mat_ldfp_recursive {{0 1}} -vec_type hip

TEST*/
