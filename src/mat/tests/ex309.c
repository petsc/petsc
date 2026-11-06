static const char help[] = "Test MATPRODUCT_AB (MatMatMult) with MATDIAGONAL and MATCONSTANTDIAGONAL against any matrix type, and MATPRODUCT_AtB (MatTransposeMatMult) of MATDIAGONAL and MATCONSTANTDIAGONAL with a dense matrix\n\n";

// Contributed by: Steven Dargaville

#include <petscmat.h>

// Compute result = X * Y (MATPRODUCT_AB) and verify it against the action of X and Y.
static PetscErrorCode CheckAB(Mat X, Mat Y, const char *what)
{
  Mat       result;
  PetscBool equal = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(MatMatMult(X, Y, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &result));
  PetscCall(MatMatMultEqual(X, Y, result, 10, &equal));
  PetscCheck(equal, PetscObjectComm((PetscObject)X), PETSC_ERR_PLIB, "MatMatMult %s gives the wrong result", what);
  PetscCall(MatDestroy(&result));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Compare C with the host reference Ref and destroy Ref
static PetscErrorCode CheckHostReference(Mat C, Mat *Ref, const char *what)
{
  Mat       Ch;
  PetscReal nrm, err;

  PetscFunctionBeginUser;
  PetscCall(MatConvert(C, MATDENSE, MAT_INITIAL_MATRIX, &Ch));
  PetscCall(MatNorm(*Ref, NORM_INFINITY, &nrm));
  PetscCall(MatAXPY(Ch, -1.0, *Ref, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(Ch, NORM_INFINITY, &err));
  PetscCheck(err <= 100.0 * PETSC_SMALL * PetscMax(nrm, 1.0), PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "MatTransposeMatMult %s gives the wrong result, error %g relative to %g", what, (double)err, (double)nrm);
  PetscCall(MatDestroy(&Ch));
  PetscCall(MatDestroy(Ref));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Check C = D^T * B = D * B for a MATDIAGONAL D and a dense B against host copies of D and B. Using host copies also
   works for a VECKOKKOS diagonal with a CUPM dense matrix, where MatMatMultEqual() fails */
static PetscErrorCode CheckDiagonalAtB(Mat D, Mat B, Mat C, const char *what)
{
  Mat                Bh;
  Vec                dg, dh;
  const PetscScalar *d;
  PetscScalar       *h;
  PetscInt           nloc;

  PetscFunctionBeginUser;
  PetscCall(MatConvert(B, MATDENSE, MAT_INITIAL_MATRIX, &Bh));
  PetscCall(MatCreateVecs(Bh, NULL, &dh));
  PetscCall(MatDiagonalGetDiagonal(D, &dg));
  PetscCall(VecGetLocalSize(dg, &nloc));
  PetscCall(VecGetArrayRead(dg, &d));
  PetscCall(VecGetArrayWrite(dh, &h));
  PetscCall(PetscArraycpy(h, d, nloc));
  PetscCall(VecRestoreArrayWrite(dh, &h));
  PetscCall(VecRestoreArrayRead(dg, &d));
  PetscCall(MatDiagonalRestoreDiagonal(D, &dg));
  PetscCall(MatDiagonalScale(Bh, dh, NULL));
  PetscCall(VecDestroy(&dh));
  PetscCall(CheckHostReference(C, &Bh, what));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Check C = CD^T * B = alpha * B for a MATCONSTANTDIAGONAL CD and a dense B against a host copy of B
static PetscErrorCode CheckConstantDiagonalAtB(Mat CD, Mat B, Mat C, const char *what)
{
  Mat         Bh;
  PetscScalar alpha;

  PetscFunctionBeginUser;
  PetscCall(MatConvert(B, MATDENSE, MAT_INITIAL_MATRIX, &Bh));
  PetscCall(MatConstantDiagonalGetConstant(CD, &alpha));
  PetscCall(MatScale(Bh, alpha));
  PetscCall(CheckHostReference(C, &Bh, what));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat           A, B, D, D2, CD, CD2, result, ref, Cr;
  Vec           dvec, rdiag, dg;
  MatType       atype;
  PetscInt      n = 6, m, ncols = 3, rstart, rend;
  PetscScalar   cval = 1.5, cval2 = 2.5;
  PetscBool     equal = PETSC_FALSE, issame = PETSC_FALSE, check_copies = PETSC_FALSE, dense_ab = PETSC_TRUE;
  PetscLogEvent event;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-check_copies", &check_copies, NULL));
  // -dense_ab 0 skips the D * B and CD * B checks, since MatMatMultEqual() fails for a VECKOKKOS diagonal with a CUPM dense matrix
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-dense_ab", &dense_ab, NULL));
  if (check_copies) PetscCall(PetscLogDefaultBegin());
  PetscCall(PetscLogEventRegister("ProductCheck", MAT_CLASSID, &event));

  // A genuinely non-diagonal sparse matrix with a parallel-safe layout; its type
  // is taken from -mat_type (e.g. aij, aijkokkos, aijcusparse, aijhipsparse).
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) {
    PetscInt    cols[2] = {i, (i + 1) % n};
    PetscScalar vals[2] = {(PetscScalar)(i + 2), 1.0};
    PetscCall(MatSetValues(A, 1, &i, 2, cols, vals, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatGetType(A, &atype));
  PetscCall(MatGetLocalSize(A, &m, NULL));

  // A dense matrix
  PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
  PetscCall(MatSetType(B, MATDENSE));
  PetscCall(MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, n, ncols));
  PetscCall(MatSetOptionsPrefix(B, "dense_"));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetUp(B));
  PetscCall(MatSetRandom(B, NULL));

  // Two MATDIAGONAL matrices from Vecs matching A's layout and VecType (so on a
  // device build with a device A everything stays on device).
  PetscCall(MatCreateVecs(A, &dvec, NULL));
  PetscCall(VecGetOwnershipRange(dvec, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) PetscCall(VecSetValue(dvec, i, (PetscScalar)(i + 3), INSERT_VALUES));
  PetscCall(VecAssemblyBegin(dvec));
  PetscCall(VecAssemblyEnd(dvec));
  PetscCall(MatCreateDiagonal(dvec, &D));
  PetscCall(VecScale(dvec, -0.5));
  PetscCall(MatCreateDiagonal(dvec, &D2));
  PetscCall(VecDestroy(&dvec));

  // Two MATCONSTANTDIAGONAL matrices.
  PetscCall(MatCreateConstantDiagonal(PETSC_COMM_WORLD, m, m, n, n, cval, &CD));
  PetscCall(MatCreateConstantDiagonal(PETSC_COMM_WORLD, m, m, n, n, cval2, &CD2));

  // MATDIAGONAL against a general (sparse) matrix, both orientations.
  PetscCall(CheckAB(A, D, "A * D (aij * MATDIAGONAL)"));
  PetscCall(CheckAB(D, A, "D * A (MATDIAGONAL * aij)"));

  // MATCONSTANTDIAGONAL against a general (sparse) matrix, both orientations.
  PetscCall(CheckAB(A, CD, "A * CD (aij * MATCONSTANTDIAGONAL)"));
  PetscCall(CheckAB(CD, A, "CD * A (MATCONSTANTDIAGONAL * aij)"));

  // MATDIAGONAL against a dense matrix
  if (dense_ab) PetscCall(CheckAB(D, B, "D * B (MATDIAGONAL * dense)"));
  PetscCall(MatTransposeMatMult(D, B, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Cr));
  PetscCall(CheckDiagonalAtB(D, B, Cr, "D^T * B (MATDIAGONAL^T * dense)"));
  PetscCall(MatDestroy(&Cr));

  // MATCONSTANTDIAGONAL against a dense matrix
  if (dense_ab) PetscCall(CheckAB(CD, B, "CD * B (MATCONSTANTDIAGONAL * dense)"));
  PetscCall(MatTransposeMatMult(CD, B, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Cr));
  PetscCall(CheckConstantDiagonalAtB(CD, B, Cr, "CD^T * B (MATCONSTANTDIAGONAL^T * dense)"));
  PetscCall(MatDestroy(&Cr));

  // MATDIAGONAL/MATCONSTANTDIAGONAL against each other.
  PetscCall(CheckAB(D, D2, "D * D (MATDIAGONAL * MATDIAGONAL)"));
  PetscCall(CheckAB(CD, CD2, "CD * CD (MATCONSTANTDIAGONAL * MATCONSTANTDIAGONAL)"));
  PetscCall(CheckAB(CD, D, "CD * D (MATCONSTANTDIAGONAL * MATDIAGONAL)"));
  PetscCall(CheckAB(D, CD, "D * CD (MATDIAGONAL * MATCONSTANTDIAGONAL)"));

  // Explicit symbolic-only phase, then numeric (mirrors callers that build the
  // product structure without an immediate numeric); verify the symbolic result
  // inherits the general operand's type, then check the numeric values.
  PetscCall(MatProductCreate(A, D, NULL, &result));
  PetscCall(MatProductSetType(result, MATPRODUCT_AB));
  PetscCall(MatProductSetFromOptions(result));
  PetscCall(MatProductSymbolic(result));
  PetscCall(PetscObjectTypeCompare((PetscObject)result, atype, &issame));
  PetscCheck(issame, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Symbolic AB product has an unexpected type");
  PetscCall(MatProductNumeric(result));
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &ref));
  PetscCall(MatDiagonalGetDiagonal(D, &rdiag));
  PetscCall(MatDiagonalScale(ref, NULL, rdiag));
  PetscCall(MatDiagonalRestoreDiagonal(D, &rdiag));
  PetscCall(MatEqual(result, ref, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Symbolic+numeric AB product does not match column scaling");
  PetscCall(MatDestroy(&ref));
  PetscCall(MatDestroy(&result));

  // MAT_REUSE_MATRIX: reuse skips the symbolic phase and re-runs only the
  // numeric, so mutating an operand and reusing must recompute the product.
  // First the anytype-output orientations (D * A, A * D, A * CD).
  PetscCall(MatMatMult(D, A, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Cr));
  PetscCall(MatMatMultEqual(D, A, Cr, 10, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "D * A (initial) gives the wrong result");
  PetscCall(MatDiagonalGetDiagonal(D, &dg));
  PetscCall(VecScale(dg, -2.5));
  PetscCall(MatDiagonalRestoreDiagonal(D, &dg));
  PetscCall(MatMatMult(D, A, MAT_REUSE_MATRIX, PETSC_DETERMINE, &Cr));
  PetscCall(MatMatMultEqual(D, A, Cr, 10, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "D * A (reuse, diagonal changed) gives the wrong result");
  PetscCall(MatDestroy(&Cr));

  PetscCall(MatMatMult(A, D, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Cr));
  PetscCall(MatDiagonalGetDiagonal(D, &dg));
  PetscCall(VecScale(dg, 0.7));
  PetscCall(MatDiagonalRestoreDiagonal(D, &dg));
  PetscCall(MatMatMult(A, D, MAT_REUSE_MATRIX, PETSC_DETERMINE, &Cr));
  PetscCall(MatMatMultEqual(A, D, Cr, 10, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "A * D (reuse, diagonal changed) gives the wrong result");
  PetscCall(MatDestroy(&Cr));

  PetscCall(MatMatMult(A, CD, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Cr));
  PetscCall(MatScale(CD, 2.0));
  PetscCall(MatMatMult(A, CD, MAT_REUSE_MATRIX, PETSC_DETERMINE, &Cr));
  PetscCall(MatMatMultEqual(A, CD, Cr, 10, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "A * CD (reuse, constant scaled) gives the wrong result");
  PetscCall(MatDestroy(&Cr));

  // Then the MATDIAGONAL/MATCONSTANTDIAGONAL-output combinations (D * D, CD * CD,
  // CD * D, D * CD); their numeric routines likewise recompute from the operands.
  PetscCall(MatMatMult(D, D2, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Cr));
  PetscCall(MatDiagonalGetDiagonal(D, &dg));
  PetscCall(VecScale(dg, 1.3));
  PetscCall(MatDiagonalRestoreDiagonal(D, &dg));
  PetscCall(MatMatMult(D, D2, MAT_REUSE_MATRIX, PETSC_DETERMINE, &Cr));
  PetscCall(MatMatMultEqual(D, D2, Cr, 10, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "D * D (reuse, diagonal changed) gives the wrong result");
  PetscCall(MatDestroy(&Cr));

  PetscCall(MatMatMult(CD, CD2, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Cr));
  PetscCall(MatScale(CD, 1.5));
  PetscCall(MatMatMult(CD, CD2, MAT_REUSE_MATRIX, PETSC_DETERMINE, &Cr));
  PetscCall(MatMatMultEqual(CD, CD2, Cr, 10, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "CD * CD (reuse, constant scaled) gives the wrong result");
  PetscCall(MatDestroy(&Cr));

  PetscCall(MatMatMult(CD, D, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Cr));
  PetscCall(MatDiagonalGetDiagonal(D, &dg));
  PetscCall(VecScale(dg, -0.9));
  PetscCall(MatDiagonalRestoreDiagonal(D, &dg));
  PetscCall(MatMatMult(CD, D, MAT_REUSE_MATRIX, PETSC_DETERMINE, &Cr));
  PetscCall(MatMatMultEqual(CD, D, Cr, 10, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "CD * D (reuse, diagonal changed) gives the wrong result");
  PetscCall(MatDestroy(&Cr));

  PetscCall(MatMatMult(D, CD, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Cr));
  PetscCall(MatScale(CD, 0.5));
  PetscCall(MatMatMult(D, CD, MAT_REUSE_MATRIX, PETSC_DETERMINE, &Cr));
  PetscCall(MatMatMultEqual(D, CD, Cr, 10, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "D * CD (reuse, constant scaled) gives the wrong result");
  PetscCall(MatDestroy(&Cr));

  // Reuse D^T * B and CD^T * B after the diagonals change and check them, then reuse them again with the copies logged
  PetscCall(MatTransposeMatMult(D, B, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Cr));
  PetscCall(MatTransposeMatMult(CD, B, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &result));
  PetscCall(MatDiagonalGetDiagonal(D, &dg));
  PetscCall(VecScale(dg, 1.7));
  PetscCall(MatDiagonalRestoreDiagonal(D, &dg));
  PetscCall(MatScale(CD, -3.0));
  PetscCall(MatTransposeMatMult(D, B, MAT_REUSE_MATRIX, PETSC_DETERMINE, &Cr));
  PetscCall(CheckDiagonalAtB(D, B, Cr, "D^T * B (reuse, diagonal changed)"));
  PetscCall(MatTransposeMatMult(CD, B, MAT_REUSE_MATRIX, PETSC_DETERMINE, &result));
  PetscCall(CheckConstantDiagonalAtB(CD, B, result, "CD^T * B (reuse, constant scaled)"));
  PetscCall(PetscLogEventBegin(event, 0, 0, 0, 0));
  PetscCall(MatTransposeMatMult(D, B, MAT_REUSE_MATRIX, PETSC_DETERMINE, &Cr));
  PetscCall(MatTransposeMatMult(CD, B, MAT_REUSE_MATRIX, PETSC_DETERMINE, &result));
  PetscCall(PetscLogEventEnd(event, 0, 0, 0, 0));
  PetscCall(CheckDiagonalAtB(D, B, Cr, "D^T * B (reuse again)"));
  PetscCall(CheckConstantDiagonalAtB(CD, B, result, "CD^T * B (reuse again)"));
  PetscCall(MatDestroy(&Cr));
  PetscCall(MatDestroy(&result));
#if PetscDefined(HAVE_DEVICE)
  // With device matrices, the logged reuse must not copy between host and device
  if (check_copies) {
    PetscEventPerfInfo info;

    PetscCall(PetscLogEventGetPerfInfo(PETSC_DETERMINE, event, &info));
    PetscCheck(info.GpuToCpuCount == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%g unexpected GPU to CPU copies (%g bytes) in MatTransposeMatMult()", info.GpuToCpuCount, info.GpuToCpuSize);
    PetscCheck(info.CpuToGpuCount == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%g unexpected CPU to GPU copies (%g bytes) in MatTransposeMatMult()", info.CpuToGpuCount, info.CpuToGpuSize);
  }
#endif

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&D));
  PetscCall(MatDestroy(&D2));
  PetscCall(MatDestroy(&CD));
  PetscCall(MatDestroy(&CD2));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  test:
    suffix: cpu
    nsize: {{1 2}}
    output_file: output/empty.out

  test:
    requires: kokkos_kernels
    suffix: kokkos
    nsize: {{1 2}}
    args: -mat_type aijkokkos
    output_file: output/empty.out

  test:
    requires: cuda
    suffix: cuda
    nsize: {{1 2}}
    args: -mat_type aijcusparse -dense_mat_type densecuda
    output_file: output/empty.out

  test:
    requires: hip
    suffix: hip
    nsize: {{1 2}}
    args: -mat_type aijhipsparse -dense_mat_type densehip
    output_file: output/empty.out

  # the copy checks on 2 processes do not need GPU-aware MPI, since these products do not communicate
  test:
    requires: cuda defined(PETSC_USE_LOG)
    suffix: cuda_copies
    nsize: {{1 2}}
    args: -mat_type aijcusparse -dense_mat_type densecuda -check_copies
    output_file: output/empty.out

  test:
    requires: hip defined(PETSC_USE_LOG)
    suffix: hip_copies
    nsize: {{1 2}}
    args: -mat_type aijhipsparse -dense_mat_type densehip -check_copies
    output_file: output/empty.out

  test:
    requires: cuda kokkos_kernels defined(PETSC_USE_LOG)
    suffix: kokkos_cuda
    nsize: {{1 2}}
    args: -mat_type aijkokkos -dense_mat_type densecuda -dense_mat_vec_type kokkos -check_copies -dense_ab 0
    output_file: output/empty.out

  test:
    requires: hip kokkos_kernels defined(PETSC_USE_LOG)
    suffix: kokkos_hip
    nsize: {{1 2}}
    args: -mat_type aijkokkos -dense_mat_type densehip -dense_mat_vec_type kokkos -check_copies -dense_ab 0
    output_file: output/empty.out
TEST*/
