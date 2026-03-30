#include <petscda.h>
#include <petsc/private/daimpl.h>
#include <petscblaslapack.h>
#include <petsc/private/daensembleimpl.h>

/*
     Code that is shared between multiple PetscDA ensemble methods including PETSCDAETKF and PETSCDALETKF

*/
/*  T-Matrix Factorization and Application Methods [Alg 6.4 line 7] */

/*
   Tolerance for matrix square root verification in debug mode
   Use a more relaxed tolerance to account for accumulated floating-point errors
   in multiple matrix operations (Y^T * T * Y involves 3 matrix multiplications).
   A tolerance of 1e-2 (1%) is reasonable for numerical verification. */
#define MATRIX_SQRT_TOLERANCE_FACTOR 1.0e-2

/*
  PetscDAEnsembleTFactor_Cholesky - Computes Cholesky factorization of T

  Input Parameters:
+ da - the PetscDA context
- S  - normalized innovation matrix (obs_size x m)

  Notes:
  Computes the lower triangular Cholesky factor L such that T = L * L^T.
  Then zeros out the upper triangular part to ensure L is strictly lower triangular.
*/
static PetscErrorCode PetscDAEnsembleTFactor_Cholesky(PetscDA da)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  PetscBLASInt      n, lda, info;
  PetscScalar      *a_array;
  PetscInt          m_T, N_T, i, j;

  PetscFunctionBegin;
  /* Initialize or update L_cholesky matrix */
  if (!en->L_cholesky) {
    PetscCall(MatDuplicate(en->I_StS, MAT_COPY_VALUES, &en->L_cholesky));
  } else {
    PetscCall(MatCopy(en->I_StS, en->L_cholesky, SAME_NONZERO_PATTERN));
  }

  /* Get matrix dimensions and convert to BLAS int */
  PetscCall(MatGetSize(en->L_cholesky, &m_T, &N_T));
  PetscCheck(m_T == N_T, PetscObjectComm((PetscObject)en->L_cholesky), PETSC_ERR_ARG_WRONG, "Matrix must be square for Cholesky");
  PetscCall(PetscBLASIntCast(N_T, &n));
  lda = n;

  /* Get array from dense matrix */
  PetscCall(MatDenseGetArrayWrite(en->L_cholesky, &a_array));

  /* Compute Cholesky factorization: A = L * L^T (lower triangular) */
  PetscCallBLAS("LAPACKpotrf", LAPACKpotrf_("L", &n, a_array, &lda, &info));
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK Cholesky factorization (xPOTRF): info=%" PetscInt_FMT ". Matrix T is not positive definite.", (PetscInt)info);

  /* Zero out upper triangular part (LAPACK leaves it unchanged) */
  for (j = 0; j < n; j++) {
    for (i = 0; i < j; i++) a_array[i + j * lda] = 0.0;
  }

  /* Restore array and finalize matrix */
  PetscCall(MatDenseRestoreArrayWrite(en->L_cholesky, &a_array));
  PetscCall(MatAssemblyBegin(en->L_cholesky, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(en->L_cholesky, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDAEnsembleTFactor_Eigen - Computes Eigendecomposition of T

  Input Parameters:
+ da - the PetscDA context
- S  - normalized innovation matrix (obs_size x m)

  Notes:
  Computes eigenvectors V and eigenvalues D such that T = V * D * V^T.
*/
static PetscErrorCode PetscDAEnsembleTFactor_Eigen(PetscDA da)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  PetscBLASInt      n, lda, lwork, info;
  PetscScalar      *a_array, *work, *eig_array;
  PetscInt          m_V, N_V;
#if defined(PETSC_USE_COMPLEX)
  PetscReal *rwork = NULL;
#endif

  PetscFunctionBegin;
  /* Initialize or update V matrix */
  if (!en->V) {
    PetscCall(MatDuplicate(en->I_StS, MAT_COPY_VALUES, &en->V));
  } else {
    PetscCall(MatCopy(en->I_StS, en->V, SAME_NONZERO_PATTERN));
  }

  /* Initialize or update eigenvalue vector */
  if (!en->sqrt_eigen_vals) PetscCall(MatCreateVecs(en->I_StS, &en->sqrt_eigen_vals, NULL));

  /* Get matrix dimensions */
  PetscCall(MatGetSize(en->V, &m_V, &N_V));
  PetscCheck(m_V == N_V, PetscObjectComm((PetscObject)en->V), PETSC_ERR_ARG_WRONG, "Matrix must be square");
  PetscCall(PetscBLASIntCast(N_V, &n));
  lda = n;

  /* Get arrays */
  PetscCall(MatDenseGetArrayWrite(en->V, &a_array));
  PetscCall(VecGetArrayWrite(en->sqrt_eigen_vals, &eig_array));

  /* Query optimal workspace size */
  lwork = -1;
  PetscCall(PetscMalloc1(1, &work));
#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscMalloc1(PetscMax(1, 3 * n - 2), &rwork));
  PetscCallBLAS("LAPACKsyev", LAPACKsyev_("V", "U", &n, a_array, &lda, (PetscReal *)eig_array, work, &lwork, rwork, &info));
#else
  PetscCallBLAS("LAPACKsyev", LAPACKsyev_("V", "U", &n, a_array, &lda, eig_array, work, &lwork, &info));
#endif
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK routine xSYEV work query: info=%" PetscInt_FMT, (PetscInt)info);

  /* Allocate workspace */
  lwork = (PetscBLASInt)PetscRealPart(work[0]);
  PetscCall(PetscFree(work));
  PetscCall(PetscMalloc1(lwork, &work));

  /* Compute eigendecomposition */
#if defined(PETSC_USE_COMPLEX)
  PetscCallBLAS("LAPACKsyev", LAPACKsyev_("V", "U", &n, a_array, &lda, (PetscReal *)eig_array, work, &lwork, rwork, &info));
  PetscCall(PetscFree(rwork));
#else
  PetscCallBLAS("LAPACKsyev", LAPACKsyev_("V", "U", &n, a_array, &lda, eig_array, work, &lwork, &info));
#endif
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK routine xSYEV: info=%" PetscInt_FMT, (PetscInt)info);

  /* Cleanup */
  PetscCall(PetscFree(work));
  PetscCall(VecRestoreArrayWrite(en->sqrt_eigen_vals, &eig_array));
  PetscCall(MatDenseRestoreArrayWrite(en->V, &a_array));

  /* Compute sqrt(eigenvalues) */
  PetscCall(VecSqrtAbs(en->sqrt_eigen_vals));

  /* Debug verification: Ensure V * D * V^T == T */
  if (PetscDefined(USE_DEBUG)) {
    PetscReal norm_T, norm_diff, relative_error;
    Mat       V_D, VDVt;

    /* Compute D * V^T by scaling rows */
    PetscCall(MatDuplicate(en->V, MAT_COPY_VALUES, &V_D));

    /* Restore D for verification (since sqrt_eigen_vals currently holds sqrt(D)) */
    PetscCall(VecPointwiseMult(en->sqrt_eigen_vals, en->sqrt_eigen_vals, en->sqrt_eigen_vals));

    PetscCall(MatDiagonalScale(V_D, NULL, en->sqrt_eigen_vals));

    /* Compute V * D * V^T */
    PetscCall(MatMatTransposeMult(V_D, en->V, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &VDVt));

    /* Compute ||V*D*V^T - T|| / ||T|| */
    PetscCall(MatAXPY(VDVt, -1.0, en->I_StS, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(en->I_StS, NORM_FROBENIUS, &norm_T));
    PetscCall(MatNorm(VDVt, NORM_FROBENIUS, &norm_diff));

    PetscCheck(norm_T > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_PLIB, "T = 0");
    relative_error = norm_diff / norm_T;
    PetscCheck(relative_error < MATRIX_SQRT_TOLERANCE_FACTOR, PetscObjectComm((PetscObject)da), PETSC_ERR_PLIB, "Eigendecomposition verification failed: ||V*D*V^T - T||/||T|| = %g", (double)relative_error);

    /* Restore sqrt(D) back to sqrt_eigen_vals */
    PetscCall(VecSqrtAbs(en->sqrt_eigen_vals));

    /* Cleanup debug matrices */
    PetscCall(MatDestroy(&V_D));
    PetscCall(MatDestroy(&VDVt));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleTFactor - Compute and store factorization of T matrix

  Collective

  Input Parameters:
+ da - the `PetscDA` context
- S  - normalized innovation matrix (obs_size x m)

  Notes:
  This function computes $T = I + S^T * S$ and stores its factorization based on
  the selected `PetscDASqrtType`.

  - For CHOLESKY mode: computes the lower triangular Cholesky factor $L$ such that $T = L * L^T$.
  - For EIGEN mode: computes eigenvectors $V$ and eigenvalues $D$ such that $T = V * D * V^T$.

  The implementation uses matrix reuse (`MAT_REUSE_MATRIX`) to minimize memory allocation
  overhead when the ensemble size remains constant across analysis cycles.

  Level: advanced

.seealso: [](ch_da), `PetscDA`, `PETSCDAETKF`, `PETSCDALETKF`, `PetscDAEnsembleApplyTInverse()`, `PetscDAEnsembleApplySqrtTInverse()`
@*/
PetscErrorCode PetscDAEnsembleTFactor(PetscDA da, Mat S)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  PetscInt          m, s_rows, s_cols;
  MatReuse          scall = MAT_INITIAL_MATRIX;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(S, MAT_CLASSID, 2);
  PetscCall(MatGetSize(S, &s_rows, &s_cols));
  m = s_cols; /* Ensemble size */
  PetscCheck(m > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Innovation matrix S must have positive columns, got %" PetscInt_FMT, m);
  PetscCheck(m == en->size, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "S matrix columns (%" PetscInt_FMT ") must match ensemble size (%" PetscInt_FMT ") defined in PetscDA", m, en->size);

  /* 2. Manage Resource Reuse */
  /* Check if we can reuse the T matrix (I_StS) and dependent factors */
  if (en->I_StS) {
    PetscInt t_rows, t_cols;
    PetscCall(MatGetSize(en->I_StS, &t_rows, &t_cols));

    /* If dimensions have changed, we must fully reallocate */
    if (t_rows != m || t_cols != m) {
      PetscCall(MatDestroy(&en->I_StS));
      PetscCall(MatDestroy(&en->V));
      PetscCall(MatDestroy(&en->L_cholesky));
      PetscCall(VecDestroy(&en->sqrt_eigen_vals));
      scall = MAT_INITIAL_MATRIX;
      PetscCall(PetscInfo(da, "Ensemble size changed (old: %" PetscInt_FMT ", new: %" PetscInt_FMT "), reallocating T matrix and factors\n", t_rows, m));
    } else {
      scall = MAT_REUSE_MATRIX;
    }
  }

  /* 3. Compute T = I + S^T * S */
  /*
     MatTransposeMatMult computes C = A^T * B (here C = S^T * S).
     When using MAT_REUSE_MATRIX, the existing C is overwritten with the new result.
  */
  PetscCall(MatTransposeMatMult(S, S, scall, PETSC_DEFAULT, &en->I_StS));

  /* Add Identity: T = (1/rho)I + S^T*S */
  PetscCall(MatShift(en->I_StS, 1.0 / en->inflation));

  /* 4. Compute Factorization based on strategy */
  switch (en->sqrt_type) {
  case PETSCDA_SQRT_CHOLESKY:
    PetscCall(PetscDAEnsembleTFactor_Cholesky(da));
    break;
  case PETSCDA_SQRT_EIGEN:
    PetscCall(PetscDAEnsembleTFactor_Eigen(da));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported PetscDA square-root type %" PetscInt_FMT, (PetscInt)en->sqrt_type);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ApplyTInverse_Cholesky - Helper for Cholesky solver path
*/
static PetscErrorCode ApplyTInverse_Cholesky(PetscDA da, Vec sdel, Vec w)
{
  PetscDA_Ensemble  *en = (PetscDA_Ensemble *)da->data;
  PetscBLASInt       n, lda, nrhs, info;
  const PetscScalar *a_array;
  PetscScalar       *b_array;
  PetscInt           m_L, N_L;

  PetscFunctionBegin;
  PetscCheck(en->L_cholesky, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Cholesky factor not computed");

  /* Get dimensions */
  PetscCall(MatGetSize(en->L_cholesky, &m_L, &N_L));
  PetscCall(PetscBLASIntCast(N_L, &n));
  lda  = n;
  nrhs = 1;

  /* Copy sdel to w for in-place solve */
  PetscCall(VecCopy(sdel, w));

  /* Get arrays */
  PetscCall(MatDenseGetArrayRead(en->L_cholesky, &a_array));
  PetscCall(VecGetArray(w, &b_array));

  /* Solve L * L^T * w = sdel using LAPACK's Cholesky solve (xPOTRS) */
  /* Note: POTRS expects the input B (w) to contain the RHS, and overwrites it with the solution */
  PetscCallBLAS("LAPACKpotrs", LAPACKpotrs_("L", &n, &nrhs, a_array, &lda, b_array, &n, &info));
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK Cholesky solve (xPOTRS): info=%" PetscInt_FMT, (PetscInt)info);

  /* Restore arrays */
  PetscCall(MatDenseRestoreArrayRead(en->L_cholesky, &a_array));
  PetscCall(VecRestoreArray(w, &b_array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ApplyTInverse_Eigen - Helper for Eigendecomposition solver path
*/
static PetscErrorCode ApplyTInverse_Eigen(PetscDA da, Vec sdel, Vec w)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  Vec               temp;

  PetscFunctionBegin;
  PetscCheck(en->V, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Eigenvectors not computed");
  PetscCheck(en->sqrt_eigen_vals, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Eigenvalues not computed");

  /* Allocate temporary vector for projection */
  PetscCall(VecDuplicate(sdel, &temp));

  /* 1. Project onto eigenvectors: temp = V^T * sdel */
  PetscCall(MatMultTranspose(en->V, sdel, temp));

  /* 2. Scale by inverse eigenvalues: temp = D^{-1} * temp */
  /* We store sqrt(D), so divide twice: temp = (temp / sqrt(D)) / sqrt(D) */
  PetscCall(VecPointwiseDivide(temp, temp, en->sqrt_eigen_vals));
  PetscCall(VecPointwiseDivide(temp, temp, en->sqrt_eigen_vals));

  /* 3. Map back to standard basis: w = V * temp */
  PetscCall(MatMult(en->V, temp, w));

  PetscCall(VecDestroy(&temp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleApplyTInverse - Apply T^{-1} to a vector [Alg 6.4 line 8]

  Collective

  Input Parameters:
+ da   - the `PetscDA` context
- sdel - input vector S^T-delta

  Output Parameter:
. w - output vector w = T^{-1} * sdel

  Notes:
  This function applies the inverse of T = I + S^T S using the stored
  factorization. For CHOLESKY mode, it uses triangular solves. For EIGEN mode,
  it uses the eigendecomposition (T^{-1} = V D^{-1} V^T).

  Level: advanced

.seealso: [](ch_da), `PetscDA`, `PETSCDAETKF`, `PETSCDALETKF`, `PetscDAEnsembleTFactor()`, `PetscDAEnsembleApplySqrtTInverse()`
@*/
PetscErrorCode PetscDAEnsembleApplyTInverse(PetscDA da, Vec sdel, Vec w)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(sdel, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(w, VEC_CLASSID, 3);

  PetscCheck(en->I_StS, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "T matrix not factored. Call PetscDAEnsembleTFactor first");

  switch (en->sqrt_type) {
  case PETSCDA_SQRT_CHOLESKY:
    PetscCall(ApplyTInverse_Cholesky(da, sdel, w));
    break;
  case PETSCDA_SQRT_EIGEN:
    PetscCall(ApplyTInverse_Eigen(da, sdel, w));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported PetscDA square-root type %" PetscInt_FMT, (PetscInt)en->sqrt_type);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ApplySqrtTInverse_Cholesky - Computes Y = L^{-T} * U using Cholesky factorization

  Notes:
  For T = L * L^T (Cholesky factorization), this computes the ASYMMETRIC square root
  T^{-1/2} = L^{-T} (upper triangular).

  This satisfies the product property:
    T^{-1/2} * (T^{-1/2})^T = L^{-T} * L^{-1} = (L * L^T)^{-1} = T^{-1}

  WARNING: L^{-T} is upper triangular and NOT symmetric. This is valid for ETKF where
  the global ensemble transform W = X_a * T^{-1/2} does not require symmetry. However,
  LETKF requires a SYMMETRIC square root T^{-1/2} = V * D^{-1/2} * V^T for the local
  ensemble perturbation update. Use PETSCDA_SQRT_EIGEN for LETKF.

  This requires solving L^T * Y = U for Y.
*/
static PetscErrorCode ApplySqrtTInverse_Cholesky(PetscDA da, Mat U, Mat Y)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscBLASInt       n, lda, nrhs, info;
  const PetscScalar *l_array;
  PetscScalar       *y_array;
  PetscInt           m_L, N_L, m_U, N_U;
  Mat                U_identity = NULL;

  PetscFunctionBegin;
  PetscCheck(en->L_cholesky, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Cholesky factor not computed");

  PetscCall(MatGetSize(en->L_cholesky, &m_L, &N_L));

  /* Handle NULL U (identity matrix case) */
  if (!U) {
    /* Create identity matrix of size m_L x m_L */
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)en->L_cholesky), PETSC_DECIDE, PETSC_DECIDE, m_L, m_L, NULL, &U_identity));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)U_identity, "dense_"));
    PetscCall(MatSetFromOptions(U_identity));
    PetscCall(MatSetUp(U_identity));
    PetscCall(MatShift(U_identity, 1.0)); /* Set diagonal to 1 */
    U = U_identity;
  }

  PetscCall(MatGetSize(U, &m_U, &N_U));
  PetscCheck(m_L == m_U, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "Cholesky factor rows (%" PetscInt_FMT ") must match U rows (%" PetscInt_FMT ")", m_L, m_U);

  PetscCall(PetscBLASIntCast(N_L, &n));
  PetscCall(PetscBLASIntCast(N_U, &nrhs));
  lda = n;

  /* Initialize Y with U for in-place solve */
  PetscCall(MatCopy(U, Y, SAME_NONZERO_PATTERN));

  /* Get direct array access */
  PetscCall(MatDenseGetArrayRead(en->L_cholesky, &l_array));
  PetscCall(MatDenseGetArrayWrite(Y, &y_array));

  /* Solve L^T * Y = U using LAPACK triangular solve (L is lower, so L^T is upper)
     TRTRS args: UPLO='L', TRANS='T', DIAG='N' */
  PetscCallBLAS("LAPACKtrtrs", LAPACKtrtrs_("L", "T", "N", &n, &nrhs, (PetscScalar *)l_array, &lda, y_array, &n, &info));
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK triangular solve (xTRTRS): info=%" PetscInt_FMT, (PetscInt)info);

  /* Restore arrays */
  PetscCall(MatDenseRestoreArrayRead(en->L_cholesky, &l_array));
  PetscCall(MatDenseRestoreArrayWrite(Y, &y_array));

  /* Cleanup temporary identity matrix if created */
  PetscCall(MatDestroy(&U_identity));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ApplySqrtTInverse_Eigen - Computes Y = V * D^{-1/2} * V^T * U.

  Notes:
  This computes the symmetric square root T^{-1/2} = V * D^{-1/2} * V^T.
  The operation is performed as Y = V * (D^{-1/2} * (V^T * U)) to strictly follow
  linear algebra operations for general matrix U.
*/
static PetscErrorCode ApplySqrtTInverse_Eigen(PetscDA da, Mat U, Mat Y)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  Mat               W;
  Vec               diag_inv;

  PetscFunctionBegin;
  PetscCheck(en->V, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Eigenvectors not computed");
  PetscCheck(en->sqrt_eigen_vals, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "Eigenvalues not computed");

  /* Prepare inverse sqrt eigenvalues: D^{-1/2}
     Note: en->sqrt_eigen_vals currently stores sqrt(D) */
  PetscCall(VecDuplicate(en->sqrt_eigen_vals, &diag_inv));
  PetscCall(VecCopy(en->sqrt_eigen_vals, diag_inv));
  PetscCall(VecReciprocal(diag_inv)); /* Now diag_inv contains 1/sqrt(D) = D^{-1/2} */

  if (U) {
    /* General case: Compute Y = V * D^{-1/2} * V^T * U */
    /* Step 1: Compute W = V^T * U (Project U onto eigenbasis) */
    PetscCall(MatTransposeMatMult(en->V, U, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &W));

    /* Step 2: Scale rows of W by D^{-1/2}: W <- D^{-1/2} * W */
    PetscCall(MatDiagonalScale(W, diag_inv, NULL));

    /* Step 3: Compute Y = V * W (Project back to standard basis)
       Y = V * (D^{-1/2} * V^T * U) */
    {
      Mat Y_temp;
      PetscCall(MatMatMult(en->V, W, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Y_temp));
      PetscCall(MatCopy(Y_temp, Y, SAME_NONZERO_PATTERN));
      PetscCall(MatDestroy(&Y_temp));
    }

    /* Cleanup */
    PetscCall(MatDestroy(&W));
  } else {
    /* U is NULL (identity): Compute Y = V * D^{-1/2} * V^T directly */
    /* Step 1: Compute W = V * D^{-1/2} (scale columns of V) */
    PetscCall(MatDuplicate(en->V, MAT_COPY_VALUES, &W));
    PetscCall(MatDiagonalScale(W, NULL, diag_inv));

    /* Step 2: Compute Y = W * V^T = V * D^{-1/2} * V^T */
    {
      Mat Y_temp;
      PetscCall(MatMatTransposeMult(W, en->V, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Y_temp));
      PetscCall(MatCopy(Y_temp, Y, SAME_NONZERO_PATTERN));
      PetscCall(MatDestroy(&Y_temp));
    }

    /* Cleanup */
    PetscCall(MatDestroy(&W));
  }

  PetscCall(VecDestroy(&diag_inv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleApplySqrtTInverse - Apply T^{-1/2} to a matrix U [Alg 6.4 line 9]

  Collective

  Input Parameters:
+ da - the `PetscDA` context
- U  - input matrix (usually Identity, but can be general)

  Output Parameter:
. Y - output matrix Y = T^{-1/2} * U

  Notes:
  This function applies the inverse square root of T = I + S^T * S using the
  stored factorization.

  - For CHOLESKY mode: Computes Y = L^{-T} U
  - For EIGEN mode: Computes Y = V D^{-1/2} V^T U

  Both results satisfy Y^T * T * Y = U^T * U, preserving the metric.

  Level: advanced

.seealso: [](ch_da), `PetscDA`, `PETSCDAETKF`, `PETSCDALETKF`, `PetscDAEnsembleTFactor()`, `PetscDAEnsembleApplyTInverse()`
@*/
PetscErrorCode PetscDAEnsembleApplySqrtTInverse(PetscDA da, Mat U, Mat Y)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  if (U) PetscValidHeaderSpecific(U, MAT_CLASSID, 2);
  PetscValidHeaderSpecific(Y, MAT_CLASSID, 3);

  PetscCheck(en->I_StS, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONGSTATE, "I_StS matrix not created. Call PetscDAEnsembleTFactor first");

  switch (en->sqrt_type) {
  case PETSCDA_SQRT_CHOLESKY:
    PetscCall(ApplySqrtTInverse_Cholesky(da, U, Y));
    break;
  case PETSCDA_SQRT_EIGEN:
    PetscCall(ApplySqrtTInverse_Eigen(da, U, Y));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported PetscDA square-root type %" PetscInt_FMT, (PetscInt)en->sqrt_type);
  }

  /* Debugging verification: Check that metric is preserved
     Verify that Y^T * T * Y = U^T * U (or Y^T * T * Y = I if U is NULL) */
  if (PetscDefined(USE_DEBUG) && U) {
    Mat       YtTY, UtU, T_Y;
    PetscReal norm_ref, norm_diff;

    /* Compute LHS: Y^T * T * Y */
    PetscCall(MatMatMult(en->I_StS, Y, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &T_Y));     /* T * Y */
    PetscCall(MatTransposeMatMult(Y, T_Y, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &YtTY)); /* Y^T * (T * Y) */

    /* Compute RHS: U^T * U */
    PetscCall(MatTransposeMatMult(U, U, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &UtU));

    /* Compute difference: Diff = LHS - RHS */
    PetscCall(MatAXPY(YtTY, -1.0, UtU, SAME_NONZERO_PATTERN));

    /* Check norms */
    PetscCall(MatNorm(UtU, NORM_FROBENIUS, &norm_ref));
    PetscCall(MatNorm(YtTY, NORM_FROBENIUS, &norm_diff));

    if (norm_ref > 0.0) PetscCheck(norm_diff / norm_ref < MATRIX_SQRT_TOLERANCE_FACTOR, PETSC_COMM_SELF, PETSC_ERR_PLIB, "T^{-1/2} verification failed. ||Y^T*T*Y - U^T*U||/||U^T*U|| = %g", (double)(norm_diff / norm_ref));

    /* Cleanup debug matrices */
    PetscCall(MatDestroy(&T_Y));
    PetscCall(MatDestroy(&YtTY));
    PetscCall(MatDestroy(&UtU));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleSetSqrtType - Selects the reduced-space square-root algorithm used during analysis.

  Logically Collective

  Input Parameters:
+ da   - the `PetscDA` object
- type - either `PETSCDA_SQRT_CHOLESKY` or `PETSCDA_SQRT_EIGEN`

  Options Database Key:
. -petscda_ensemble_sqrt_type <cholesky or eigen> - set the `PetscDASqrtType`

  Level: advanced

.seealso: [](ch_da), `PetscDA`, `PETSCDAETKF`, `PETSCDALETKF`, `PetscDASqrtType`, `PetscDAEnsembleGetSqrtType()`
@*/
PetscErrorCode PetscDAEnsembleSetSqrtType(PetscDA da, PetscDASqrtType type)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscCheck(type == PETSCDA_SQRT_CHOLESKY || type == PETSCDA_SQRT_EIGEN, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Invalid PetscDA square-root type %" PetscInt_FMT, (PetscInt)type);

  en->sqrt_type = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleGetSqrtType - Retrieves the current square-root implementation configured for analysis.

  Not Collective

  Input Parameters:
. da - the `PetscDA` object

  Output Parameter:
. type - on output, the configured `PetscDASqrtType`

  Level: advanced

.seealso: [](ch_da), `PetscDA`, `PETSCDAETKF`, `PETSCDALETKF`, `PetscDAEnsembleSetSqrtType()`
@*/
PetscErrorCode PetscDAEnsembleGetSqrtType(PetscDA da, PetscDASqrtType *type)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(type, 2);
  *type = en->sqrt_type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleSetInflation - Sets the inflation factor for the data assimilation method.

  Logically Collective

  Input Parameters:
+ da        - the `PetscDA` context
- inflation - the inflation factor (must be >= 1.0)

  Level: intermediate

.seealso: [](ch_da), `PetscDA`, `PETSCDAETKF`, `PETSCDALETKF`, `PetscDAEnsembleGetInflation()`
@*/
PetscErrorCode PetscDAEnsembleSetInflation(PetscDA da, PetscReal inflation)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidLogicalCollectiveReal(da, inflation, 2);
  PetscCheck(inflation >= 1.0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Inflation factor must be >= 1.0, got %g", (double)inflation);
  en->inflation = inflation;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleGetInflation - Gets the inflation factor for the data assimilation method.

  Not Collective

  Input Parameter:
. da - the `PetscDA` context

  Output Parameter:
. inflation - the inflation factor

  Level: intermediate

.seealso: [](ch_da), `PetscDA`, `PETSCDAETKF`, `PETSCDALETKF`, `PetscDAEnsembleSetInflation()`
@*/
PetscErrorCode PetscDAEnsembleGetInflation(PetscDA da, PetscReal *inflation)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(inflation, 2);
  *inflation = en->inflation;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleGetMember - Returns a read-only view of an ensemble member stored in the `PetscDA`.

  Collective

  Input Parameters:
+ da         - the `PetscDA` context
- member_idx - index of the requested member (0 <= idx < ensemble_size)

  Output Parameter:
. member - read-only vector view; call `PetscDAEnsembleRestoreMember()` when done

  Level: intermediate

.seealso: [](ch_da), `PetscDA`, `PETSCDAETKF`, `PETSCDALETKF`, `PetscDAEnsembleRestoreMember()`, `PetscDAEnsembleSetMember()`
@*/
PetscErrorCode PetscDAEnsembleGetMember(PetscDA da, PetscInt member_idx, Vec *member)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(member, 3);
  PetscCheck(en->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "PetscDASetUp() must be called before accessing ensemble members");
  PetscCheck(member_idx >= 0 && member_idx < en->size, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Member index %" PetscInt_FMT " out of range [0, %" PetscInt_FMT ")", member_idx, en->size);

  PetscCall(MatDenseGetColumnVecRead(en->ensemble, member_idx, member));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleRestoreMember - Returns a column view obtained with `PetscDAEnsembleGetMember()`.

  Collective

  Input Parameters:
+ da         - the `PetscDA` context
. member_idx - index that was previously requested
- member     - location that holds the view to restore

  Level: intermediate

.seealso: [](ch_da), `PetscDA`, `PETSCDAETKF`, `PETSCDALETKF`, `PetscDAEnsembleGetMember()`
@*/
PetscErrorCode PetscDAEnsembleRestoreMember(PetscDA da, PetscInt member_idx, Vec *member)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(member, 3);
  PetscCheck(member_idx >= 0 && member_idx < en->size, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Member index %" PetscInt_FMT " out of range [0, %" PetscInt_FMT ")", member_idx, en->size);

  PetscCall(MatDenseRestoreColumnVecRead(en->ensemble, member_idx, member));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleSetMember - Overwrites an ensemble member with user-provided state data.

  Collective

  Input Parameters:
+ da         - the `PetscDA` context
. member_idx - index of the entry to modify
- member     - vector containing the new state values

  Level: intermediate

.seealso: [](ch_da), `PetscDA`, `PETSCDAETKF`, `PETSCDALETKF`, `PetscDAEnsembleGetMember()`
@*/
PetscErrorCode PetscDAEnsembleSetMember(PetscDA da, PetscInt member_idx, Vec member)
{
  Vec               col;
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(member, VEC_CLASSID, 3);
  PetscCheck(en->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "PetscDASetUp() must be called before setting ensemble members");
  PetscCheck(member_idx >= 0 && member_idx < en->size, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Member index %" PetscInt_FMT " out of range [0, %" PetscInt_FMT ")", member_idx, en->size);

  PetscCall(MatDenseGetColumnVecWrite(en->ensemble, member_idx, &col));
  PetscCall(VecCopy(member, col));
  PetscCall(MatDenseRestoreColumnVecWrite(en->ensemble, member_idx, &col));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleComputeMean - Computes ensemble mean for a `PetscDA`

  Collective

  Input Parameter:
. da - the `PetscDA` context

  Output Parameter:
. mean - vector that will hold the ensemble mean

  Level: intermediate

.seealso: [](ch_da), `PetscDA`, `PETSCDAETKF`, `PETSCDALETKF`, `PetscDAEnsembleComputeAnomalies()`
@*/
PetscErrorCode PetscDAEnsembleComputeMean(PetscDA da, Vec mean)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  PetscScalar       inv_m;
  PetscInt          m;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(mean, VEC_CLASSID, 2);
  PetscCheck(en->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "PetscDASetUp() must be called before computing the ensemble mean");
  PetscCheck(en->size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_WRONG, "Ensemble size must be positive");

  m     = en->size;
  inv_m = 1.0 / (PetscScalar)m;
  PetscCall(MatGetRowSum(en->ensemble, mean));
  PetscCall(VecScale(mean, inv_m));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleInitialize - Initialize ensemble members with Gaussian perturbations

  Input Parameters:
+ da            - PetscDA context
. x0            - Background state
. obs_error_std - Standard deviation for perturbations
- rng           - Random number generator

  Level: beginner

  Notes:
  Each ensemble member is initialized as x0 + Gaussian(0, obs_error_std)

.seealso: [](ch_da), `PETSCDAETKF`, `PETSCDALETKF`, `PetscDA`
@*/
PetscErrorCode PetscDAEnsembleInitialize(PetscDA da, Vec x0, PetscReal obs_error_std, PetscRandom rng)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  Vec       member, col, x_mean;
  PetscInt  i;
  PetscReal scale;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(x0, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(rng, PETSC_RANDOM_CLASSID, 4);
  PetscCall(VecDuplicate(x0, &member));
  PetscCall(VecDuplicate(x0, &x_mean));

  /*
     Scale factor to maintain consistent ensemble spread across different ensemble sizes.
     After removing the sample mean, the ensemble variance is approximately:
       Var_final ~= Var_initial * (m-1)/m
     To maintain consistent initial spread regardless of m, we scale by sqrt(m/(m-1)).
     This ensures the final ensemble spread is approximately obs_error_std^2. */
  scale = PetscSqrtReal((PetscReal)en->size / (PetscReal)(en->size - 1));

  /* Populate the Gaussian draws with scaled standard deviation */
  for (i = 0; i < en->size; i++) {
    PetscCall(VecSetRandomGaussian(member, rng, 0.0, obs_error_std * scale));
    PetscCall(PetscDAEnsembleSetMember(da, i, member));
  }
  /* get mean of perturbations */
  PetscCall(PetscDAEnsembleComputeMean(da, x_mean));
  /* remove mean and add x0 */
  for (i = 0; i < en->size; i++) {
    PetscCall(MatDenseGetColumnVecWrite(en->ensemble, i, &col));
    PetscCall(VecAXPY(col, -1.0, x_mean));
    PetscCall(VecAXPY(col, 1.0, x0));
    PetscCall(MatDenseRestoreColumnVecWrite(en->ensemble, i, &col));
  }

  PetscCall(VecDestroy(&member));
  PetscCall(VecDestroy(&x_mean));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleComputeAnomalies - Forms the state-space anomalies matrix for a `PetscDA`.

  Collective

  Input Parameters:
+ da      - the `PetscDA` context
- mean_in - optional mean state vector (pass `NULL` to compute internally)

  Output Parameter:
. anomalies_out - location to store the newly created anomalies matrix

  Notes:
  If `mean` is `NULL`, the function will create a temporary vector and compute
  the ensemble mean using `PetscDAEnsembleComputeMean()`. If `mean` is provided,
  it will be used directly, which can improve performance when the mean has
  already been computed.

  Level: intermediate

.seealso: [](ch_da), `PetscDA`, `PETSCDAETKF`, `PETSCDALETKF`, `PetscDAEnsembleComputeMean()`
@*/
PetscErrorCode PetscDAEnsembleComputeAnomalies(PetscDA da, Vec mean_in, Mat *anomalies_out)
{
  PetscDA_Ensemble *en   = (PetscDA_Ensemble *)da->data;
  Vec               mean = NULL;
  Vec               col_in, col_out;
  Mat               anomalies;
  MPI_Comm          comm;
  PetscReal         scale;
  PetscInt          ensemble_size;
  PetscInt          j;
  PetscBool         mean_created = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  if (mean_in) PetscValidHeaderSpecific(mean_in, VEC_CLASSID, 2);
  PetscAssertPointer(anomalies_out, 3);
  PetscCheck(en->ensemble, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "PetscDASetUp() must be called before computing anomalies");
  PetscCheck(en->size > 1, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be at least 2 to form anomalies");
  PetscCheck(da->state_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "State size must be positive");

  /* Cache frequently-used values for clarity and efficiency */
  ensemble_size = en->size;
  comm          = PetscObjectComm((PetscObject)en->ensemble);

  /*
    Compute normalization scale for anomalies.
    Alg 6.4 line 2: anomalies are normalized by 1/sqrt(m-1) so that
    the anomalies matrix X satisfies X*X^T = ensemble covariance matrix.
    This ensures proper statistical properties for ensemble-based methods.
  */
  scale = 1.0 / PetscSqrtReal((PetscReal)(ensemble_size - 1));

  /* Allocate anomalies matrix (state_size x ensemble_size) */
  PetscCall(MatCreateDense(comm, da->local_state_size, PETSC_DECIDE, da->state_size, ensemble_size, NULL, &anomalies));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)anomalies, "dense_"));
  PetscCall(MatSetFromOptions(anomalies));
  PetscCall(MatSetUp(anomalies));

  /* Use provided mean or create and compute it */
  if (mean_in) {
    mean = mean_in;
  } else {
    /* Create and compute ensemble mean vector */
    PetscCall(MatCreateVecs(anomalies, NULL, &mean));
    PetscCall(VecSetFromOptions(mean));
    mean_created = PETSC_TRUE;

    /* Alg 6.4 line 1: \bar{x} = (1/m)\sum_j x^{(j)} */
    PetscCall(PetscDAEnsembleComputeMean(da, mean));
  }

  /*
    Form anomalies by subtracting mean from each ensemble member and scaling.
    For each column j: anomaly_j = (ensemble_j - mean) / sqrt(m-1)
  */
  for (j = 0; j < ensemble_size; ++j) {
    PetscCall(MatDenseGetColumnVecRead(en->ensemble, j, &col_in));
    PetscCall(MatDenseGetColumnVecWrite(anomalies, j, &col_out));

    /* Alg 6.4 line 2: subtract the mean column-wise to form x^{(j)} - \bar{x} */
    PetscCall(VecWAXPY(col_out, -1.0, mean, col_in));
    /* Alg 6.4 line 2: scale anomalies by 1/\sqrt{m-1} */
    PetscCall(VecScale(col_out, scale));

    PetscCall(MatDenseRestoreColumnVecWrite(anomalies, j, &col_out));
    PetscCall(MatDenseRestoreColumnVecRead(en->ensemble, j, &col_in));
  }
  /* Transfer ownership to output and clean up temporary resources */
  *anomalies_out = anomalies;
  if (mean_created) PetscCall(VecDestroy(&mean));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleAnalysis - Executes the analysis (update) step using sparse observation matrix H

  Collective

  Input Parameters:
+ da          - the `PetscDA` context
. observation - observation vector y in R^P
- H           - observation operator matrix (P x N), sparse AIJ format

  Notes:
  The observation matrix H maps from state space (N dimensions) to observation
  space (P dimensions): y = H*x + noise

  H must be a sparse AIJ matrix

  For identity observations (observe entire state), use an identity matrix for H.
  For partial observations, set appropriate rows and columns to observe
  specific state components.

  Level: intermediate

.seealso: [](ch_da), `PetscDA`, `PETSCDAETKF`, `PETSCDALETKF`, `PetscDAEnsembleForecast()`, `PetscDASetObsErrorVariance()`
@*/
PetscErrorCode PetscDAEnsembleAnalysis(PetscDA da, Vec observation, Mat H)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  PetscInt          h_rows, h_cols;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidHeaderSpecific(observation, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(H, MAT_CLASSID, 3);
  PetscCheck(en->size > 1, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size must be > 1, got %" PetscInt_FMT, en->size);
  PetscCall(MatGetSize(H, &h_rows, &h_cols));
  PetscCheck(h_rows == da->obs_size, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "H matrix rows (%" PetscInt_FMT ") must match obs_size (%" PetscInt_FMT ")", h_rows, da->obs_size);
  PetscCheck(h_cols == da->state_size, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "H matrix cols (%" PetscInt_FMT ") must match state_size (%" PetscInt_FMT ")", h_cols, da->state_size);
  PetscCall(VecGetSize(observation, &h_rows));
  PetscCheck(h_rows == da->obs_size, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_INCOMP, "observation vector size (%" PetscInt_FMT ") must match obs_size (%" PetscInt_FMT ")", h_rows, da->obs_size);

  PetscCall(PetscLogEventBegin(PetscDA_Analysis, (PetscObject)da, 0, 0, 0));
  PetscCall((*en->analysis)(da, observation, H));
  PetscCall(PetscLogEventEnd(PetscDA_Analysis, (PetscObject)da, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDAEnsembleForecast - Advances every ensemble member through the user-supplied forecast model.

  Collective

  Input Parameters:
+ da    - the `PetscDA` context
. model - routine that evaluates the model map `f(input, output; ctx)`
- ctx   - optional context for `model`

  Level: intermediate

.seealso: [](ch_da), `PetscDA`, `PETSCDAETKF`, `PETSCDALETKF`, `PetscDAEnsembleAnalysis()`
@*/
PetscErrorCode PetscDAEnsembleForecast(PetscDA da, PetscErrorCode (*model)(Vec, Vec, PetscCtx), PetscCtx ctx)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscCall((*en->forecast)(da, model, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDAView_Ensemble(PetscDA da, PetscViewer viewer)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  PetscBool         iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Ensemble size: %" PetscInt_FMT "\n", en->size));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Assembled: %s\n", en->assembled ? "true" : "false"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Inflation: %g\n", (double)en->inflation));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Square root type: %s\n", (en->sqrt_type == PETSCDA_SQRT_EIGEN) ? "eigen" : "cholesky"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDASetUp_Ensemble(PetscDA da)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  MPI_Comm          comm;

  PetscFunctionBegin;
  if (en->assembled) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCheck(da->state_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "Must set state size before calling PetscDASetUp()");
  PetscCheck(da->obs_size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "Must set observation size before calling PetscDASetUp()");
  PetscCheck(en->size > 0, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "Must set ensemble size before calling PetscDASetUp()");

  comm = PetscObjectComm((PetscObject)da);
  if (!en->ensemble) {
    PetscCall(MatCreateDense(comm, da->local_state_size, PETSC_DECIDE, da->state_size, en->size, NULL, &en->ensemble));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)en->ensemble, "dense_"));
    PetscCall(MatSetFromOptions(en->ensemble));
    PetscCall(MatSetUp(en->ensemble));
  }
  en->assembled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleSetSize - Sets the ensemble dimensions used by a `PetscDA`.

  Collective

  Input Parameters:
+ da            - the `PetscDA` context
- ensemble_size - number of ensemble members

  Options Database Key:
. -petscda_ensemble_size <size> - number of ensemble members

  Level: beginner

  Note:
  The size must be greater than or equal to two. See the scale factor in `PetscDAEnsembleInitialize()` and `PetscDALETKFLocalAnalysis()`

.seealso: [](ch_da), `PetscDA`, `PETSCDAETKF`, `PETSCDALETKF`, `PetscDAGetSizes()`, `PetscDASetSizes()`, `PetscDASetUp()`
@*/
PetscErrorCode PetscDAEnsembleSetSize(PetscDA da, PetscInt ensemble_size)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscValidLogicalCollectiveInt(da, ensemble_size, 2);
  PetscCheck(!en->assembled, PetscObjectComm((PetscObject)da), PETSC_ERR_ORDER, "Cannot change sizes after PetscDASetUp() has been called");
  PetscCheck(ensemble_size > 1, PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_SIZ, "Ensemble size must be at least two");
  en->size = ensemble_size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleGetSize - Retrieves the dimension of the ensemble in a `PetscDA`.

  Not Collective

  Input Parameter:
. da - the `PetscDA` context

  Output Parameters:
. ensemble_size - number of ensemble members

  Level: beginner

.seealso: [](ch_da), `PetscDA`, `PETSCDAETKF`, `PETSCDALETKF`, `PetscDASetSizes()`, `PetscDAGetSizes()`
@*/
PetscErrorCode PetscDAEnsembleGetSize(PetscDA da, PetscInt *ensemble_size)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, PETSCDA_CLASSID, 1);
  PetscAssertPointer(ensemble_size, 2);
  *ensemble_size = en->size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDASetFromOptions_Ensemble(PetscDA da, PetscOptionItems *PetscOptionsObjectPtr)
{
  PetscDA_Ensemble *en                 = (PetscDA_Ensemble *)da->data;
  PetscOptionItems  PetscOptionsObject = *PetscOptionsObjectPtr;
  char              sqrt_type_name[256];
  PetscBool         sqrt_set = PETSC_FALSE, flg;
  const char       *sqrt_default;
  PetscDASqrtType   sqrt_type;
  PetscReal         inflation_val = en->inflation;
  PetscBool         inflation_set;
  PetscInt          ensemble_size;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PetscDA Ensemble Options");

  PetscCall(PetscOptionsReal("-petscda_ensemble_inflation", "Inflation factor", "PetscDAEnsembleSetInflation", en->inflation, &inflation_val, &inflation_set));
  if (inflation_set) PetscCall(PetscDAEnsembleSetInflation(da, inflation_val));

  sqrt_default = (en->sqrt_type == PETSCDA_SQRT_EIGEN) ? "eigen" : "cholesky";
  PetscCall(PetscOptionsString("-petscda_ensemble_sqrt_type", "Matrix square root factorization", "PetscDASetSqrtType", sqrt_default, sqrt_type_name, sizeof(sqrt_type_name), &sqrt_set));
  if (sqrt_set) {
    PetscBool match_cholesky, match_eigen;
    PetscCall(PetscStrcmp(sqrt_type_name, "cholesky", &match_cholesky));
    PetscCall(PetscStrcmp(sqrt_type_name, "eigen", &match_eigen));
    if (match_cholesky) {
      sqrt_type = PETSCDA_SQRT_CHOLESKY;
    } else if (match_eigen) {
      sqrt_type = PETSCDA_SQRT_EIGEN;
    } else SETERRQ(PetscObjectComm((PetscObject)da), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscDA square-root type \"%s\"", sqrt_type_name);
    PetscCall(PetscDAEnsembleSetSqrtType(da, sqrt_type));
  }
  PetscCall(PetscOptionsInt("-petscda_ensemble_size", "Number of ensemble members", "PetscDAEnsembleSetSize", en->size, &ensemble_size, &flg));
  if (flg) PetscCall(PetscDAEnsembleSetSize(da, ensemble_size));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDADestroy_Ensemble(PetscDA da)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&en->ensemble));
  PetscCall(VecDestroy(&da->obs_error_var));
  PetscCall(MatDestroy(&da->R));

  /* Destroy T-matrix factorization data */
  PetscCall(MatDestroy(&en->V));
  PetscCall(MatDestroy(&en->L_cholesky));
  PetscCall(VecDestroy(&en->sqrt_eigen_vals));
  PetscCall(MatDestroy(&en->I_StS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDACreate_Ensemble(PetscDA da)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;

  PetscFunctionBegin;
  en->size      = 0;
  en->ensemble  = NULL;
  en->assembled = PETSC_FALSE;
  en->inflation = 1.0;

  /* Initialize T-matrix factorization fields */
  en->sqrt_type       = PETSCDA_SQRT_EIGEN;
  en->V               = NULL;
  en->L_cholesky      = NULL;
  en->sqrt_eigen_vals = NULL;
  en->I_StS           = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDAEnsembleComputeNormalizedInnovationMatrix - Computes S = R^{-1/2}(Z - y_mean * 1')/sqrt(m-1) [Alg 6.4 line 5]

  Collective

  Input Parameters:
+ Z          - observation ensemble matrix
. y_mean     - mean of observations
. r_inv_sqrt - R^{-1/2}
. m          - ensemble size
- scale      - 1/sqrt(m-1)

  Output Parameter:
. S - normalized innovation matrix

  Level: developer

.seealso: [](ch_da), `PetscDA`, `PETSCDAETKF`, `PETSCDALETKF`, `PetscDASetSizes()`, `PetscDAGetSizes()`
@*/
PetscErrorCode PetscDAEnsembleComputeNormalizedInnovationMatrix(Mat Z, Vec y_mean, Vec r_inv_sqrt, PetscInt m, PetscScalar scale, Mat S)
{
  const PetscScalar *z_array, *y_array, *r_array;
  PetscScalar       *s_array;
  PetscInt           obs_size, obs_size_local, z_cols, i, j;
  PetscInt           y_local_size, r_local_size;
  PetscInt           lda_z, lda_s;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Z, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(y_mean, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(r_inv_sqrt, VEC_CLASSID, 3);
  PetscValidLogicalCollectiveInt(Z, m, 4);
  PetscValidLogicalCollectiveScalar(Z, scale, 5);
  PetscValidHeaderSpecific(S, MAT_CLASSID, 6);
  PetscCheck(m > 0, PetscObjectComm((PetscObject)Z), PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size m must be positive, got %" PetscInt_FMT, m);
  PetscCall(MatGetSize(Z, &obs_size, &z_cols));
  PetscCall(MatGetLocalSize(Z, &obs_size_local, NULL));
  PetscCheck(z_cols == m, PetscObjectComm((PetscObject)Z), PETSC_ERR_ARG_INCOMP, "Matrix Z has %" PetscInt_FMT " columns but ensemble size is %" PetscInt_FMT, z_cols, m);

  /* Verify vector dimensions match observation size (both global and local) */
  PetscCall(VecGetLocalSize(y_mean, &y_local_size));
  PetscCall(VecGetLocalSize(r_inv_sqrt, &r_local_size));
  PetscCheck(y_local_size == obs_size_local, PetscObjectComm((PetscObject)Z), PETSC_ERR_ARG_INCOMP, "Vector y_mean local size %" PetscInt_FMT " does not match matrix local rows %" PetscInt_FMT, y_local_size, obs_size_local);
  PetscCheck(r_local_size == obs_size_local, PetscObjectComm((PetscObject)Z), PETSC_ERR_ARG_INCOMP, "Vector r_inv_sqrt local size %" PetscInt_FMT " does not match matrix local rows %" PetscInt_FMT, r_local_size, obs_size_local);

  /* Get direct access to arrays for performance */
  PetscCall(MatDenseGetArrayRead(Z, &z_array));
  PetscCall(MatDenseGetArrayWrite(S, &s_array));
  PetscCall(VecGetArrayRead(y_mean, &y_array));
  PetscCall(VecGetArrayRead(r_inv_sqrt, &r_array));

  /* Get Leading Dimension (LDA) to handle padding/strides correctly */
  PetscCall(MatDenseGetLDA(Z, &lda_z));
  PetscCall(MatDenseGetLDA(S, &lda_s));

  /* Compute normalized innovation: S_ij = (Z_ij - y_mean_i) * scale * r_inv_sqrt_i
     Iterate column-wise (j) then row-wise (i) for optimal cache access with column-major storage */
  for (j = 0; j < m; j++) {
    const PetscScalar *z_col = z_array + j * lda_z;
    PetscScalar       *s_col = s_array + j * lda_s;

    for (i = 0; i < obs_size_local; i++) s_col[i] = (z_col[i] - y_array[i]) * scale * r_array[i];
  }

  /* Restore arrays */
  PetscCall(VecRestoreArrayRead(r_inv_sqrt, &r_array));
  PetscCall(VecRestoreArrayRead(y_mean, &y_array));
  PetscCall(MatDenseRestoreArrayWrite(S, &s_array));
  PetscCall(MatDenseRestoreArrayRead(Z, &z_array));
  PetscFunctionReturn(PETSC_SUCCESS);
}
