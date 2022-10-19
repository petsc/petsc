static char help[] = "Tests MATH2OPUS\n\n";

#include <petscmat.h>
#include <petscsf.h>

static PetscScalar GenEntry_Symm(PetscInt sdim, PetscReal x[], PetscReal y[], void *ctx)
{
  PetscInt  d;
  PetscReal clength = sdim == 3 ? 0.2 : 0.1;
  PetscReal dist, diff = 0.0;

  for (d = 0; d < sdim; d++) diff += (x[d] - y[d]) * (x[d] - y[d]);
  dist = PetscSqrtReal(diff);
  return PetscExpReal(-dist / clength);
}

static PetscScalar GenEntry_Unsymm(PetscInt sdim, PetscReal x[], PetscReal y[], void *ctx)
{
  PetscInt  d;
  PetscReal clength = sdim == 3 ? 0.2 : 0.1;
  PetscReal dist, diff = 0.0, nx = 0.0, ny = 0.0;

  for (d = 0; d < sdim; d++) nx += x[d] * x[d];
  for (d = 0; d < sdim; d++) ny += y[d] * y[d];
  for (d = 0; d < sdim; d++) diff += (x[d] - y[d]) * (x[d] - y[d]);
  dist = PetscSqrtReal(diff);
  return nx > ny ? PetscExpReal(-dist / clength) : PetscExpReal(-dist / clength) + 1.;
}

int main(int argc, char **argv)
{
  Mat          A, B, C, D;
  Vec          v, x, y, Ax, Ay, Bx, By;
  PetscRandom  r;
  PetscLayout  map;
  PetscScalar *Adata = NULL, *Cdata = NULL, scale = 1.0;
  PetscReal   *coords, nA, nD, nB, err, nX, norms[3];
  PetscInt     N, n = 64, dim = 1, i, j, nrhs = 11, lda = 0, ldc = 0, ldu = 0, nlr = 7, nt, ntrials = 2;
  PetscMPIInt  size, rank;
  PetscBool    testlayout = PETSC_FALSE, flg, symm = PETSC_FALSE, Asymm = PETSC_TRUE, kernel = PETSC_TRUE;
  PetscBool    checkexpl = PETSC_FALSE, agpu = PETSC_FALSE, bgpu = PETSC_FALSE, cgpu = PETSC_FALSE, flgglob;
  PetscBool    testtrans, testnorm, randommat = PETSC_TRUE, testorthog, testcompress, testhlru;
  void (*approxnormfunc)(void);
  void (*Anormfunc)(void);

#if defined(PETSC_HAVE_MPI_INIT_THREAD)
  PETSC_MPI_THREAD_REQUIRED = MPI_THREAD_MULTIPLE;
#endif
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ng", &N, &flgglob));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nrhs", &nrhs, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-lda", &lda, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ldc", &ldc, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nlr", &nlr, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ldu", &ldu, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-matmattrials", &ntrials, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-randommat", &randommat, NULL));
  if (!flgglob) PetscCall(PetscOptionsGetBool(NULL, NULL, "-testlayout", &testlayout, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-Asymm", &Asymm, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-symm", &symm, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-kernel", &kernel, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-checkexpl", &checkexpl, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-agpu", &agpu, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-bgpu", &bgpu, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-cgpu", &cgpu, NULL));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-scale", &scale, NULL));
  if (!Asymm) symm = PETSC_FALSE;

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  /* Disable tests for unimplemented variants */
  testtrans    = (PetscBool)(size == 1 || symm);
  testnorm     = (PetscBool)(size == 1 || symm);
  testorthog   = (PetscBool)(size == 1 || symm);
  testcompress = (PetscBool)(size == 1 || symm);
  testhlru     = (PetscBool)(size == 1);

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscLayoutCreate(PETSC_COMM_WORLD, &map));
  if (testlayout) {
    if (rank % 2) n = PetscMax(2 * n - 5 * rank, 0);
    else n = 2 * n + rank;
  }
  if (!flgglob) {
    PetscCall(PetscLayoutSetLocalSize(map, n));
    PetscCall(PetscLayoutSetUp(map));
    PetscCall(PetscLayoutGetSize(map, &N));
  } else {
    PetscCall(PetscLayoutSetSize(map, N));
    PetscCall(PetscLayoutSetUp(map));
    PetscCall(PetscLayoutGetLocalSize(map, &n));
  }
  PetscCall(PetscLayoutDestroy(&map));

  if (lda) PetscCall(PetscMalloc1(N * (n + lda), &Adata));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, n, n, N, N, Adata, &A));
  PetscCall(MatDenseSetLDA(A, n + lda));

  /* Create random points; these are replicated in order to populate a dense matrix and to compare sequential and dense runs
     The constructor for MATH2OPUS can take as input the distributed coordinates and replicates them internally in case
     a kernel construction is requested */
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &r));
  PetscCall(PetscRandomSetFromOptions(r));
  PetscCall(PetscRandomSetSeed(r, 123456));
  PetscCall(PetscRandomSeed(r));
  PetscCall(PetscMalloc1(N * dim, &coords));
  PetscCall(PetscRandomGetValuesReal(r, N * dim, coords));
  PetscCall(PetscRandomDestroy(&r));

  if (kernel || !randommat) {
    MatH2OpusKernel k = Asymm ? GenEntry_Symm : GenEntry_Unsymm;
    PetscInt        ist, ien;

    PetscCall(MatGetOwnershipRange(A, &ist, &ien));
    for (i = ist; i < ien; i++) {
      for (j = 0; j < N; j++) PetscCall(MatSetValue(A, i, j, (*k)(dim, coords + i * dim, coords + j * dim, NULL), INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    if (kernel) {
      PetscCall(MatCreateH2OpusFromKernel(PETSC_COMM_WORLD, n, n, N, N, dim, coords + ist * dim, PETSC_TRUE, k, NULL, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, &B));
    } else {
      PetscCall(MatCreateH2OpusFromMat(A, dim, coords + ist * dim, PETSC_TRUE, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, &B));
    }
  } else {
    PetscInt ist;

    PetscCall(MatGetOwnershipRange(A, &ist, NULL));
    PetscCall(MatSetRandom(A, NULL));
    if (Asymm) {
      PetscCall(MatTranspose(A, MAT_INITIAL_MATRIX, &B));
      PetscCall(MatAXPY(A, 1.0, B, SAME_NONZERO_PATTERN));
      PetscCall(MatDestroy(&B));
      PetscCall(MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE));
    }
    PetscCall(MatCreateH2OpusFromMat(A, dim, coords + ist * dim, PETSC_TRUE, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, &B));
  }
  PetscCall(PetscFree(coords));
  if (agpu) PetscCall(MatConvert(A, MATDENSECUDA, MAT_INPLACE_MATRIX, &A));
  PetscCall(MatViewFromOptions(A, NULL, "-A_view"));

  PetscCall(MatSetOption(B, MAT_SYMMETRIC, symm));

  /* assemble the H-matrix */
  PetscCall(MatBindToCPU(B, (PetscBool)!bgpu));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(B, NULL, "-B_view"));

  /* Test MatScale */
  PetscCall(MatScale(A, scale));
  PetscCall(MatScale(B, scale));

  /* Test MatMult */
  PetscCall(MatCreateVecs(A, &Ax, &Ay));
  PetscCall(MatCreateVecs(B, &Bx, &By));
  PetscCall(VecSetRandom(Ax, NULL));
  PetscCall(VecCopy(Ax, Bx));
  PetscCall(MatMult(A, Ax, Ay));
  PetscCall(MatMult(B, Bx, By));
  PetscCall(VecViewFromOptions(Ay, NULL, "-mult_vec_view"));
  PetscCall(VecViewFromOptions(By, NULL, "-mult_vec_view"));
  PetscCall(VecNorm(Ay, NORM_INFINITY, &nX));
  PetscCall(VecAXPY(Ay, -1.0, By));
  PetscCall(VecViewFromOptions(Ay, NULL, "-mult_vec_view"));
  PetscCall(VecNorm(Ay, NORM_INFINITY, &err));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatMult err %g\n", err / nX));
  PetscCall(VecScale(By, -1.0));
  PetscCall(MatMultAdd(B, Bx, By, By));
  PetscCall(VecNorm(By, NORM_INFINITY, &err));
  PetscCall(VecViewFromOptions(By, NULL, "-mult_vec_view"));
  if (err > 10. * PETSC_SMALL) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatMultAdd err %g\n", err));

  /* Test MatNorm */
  PetscCall(MatNorm(A, NORM_INFINITY, &norms[0]));
  PetscCall(MatNorm(A, NORM_1, &norms[1]));
  norms[2] = -1.; /* NORM_2 not supported */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "A Matrix norms:        infty=%g, norm_1=%g, norm_2=%g\n", (double)norms[0], (double)norms[1], (double)norms[2]));
  PetscCall(MatGetOperation(A, MATOP_NORM, &Anormfunc));
  PetscCall(MatGetOperation(B, MATOP_NORM, &approxnormfunc));
  PetscCall(MatSetOperation(A, MATOP_NORM, approxnormfunc));
  PetscCall(MatNorm(A, NORM_INFINITY, &norms[0]));
  PetscCall(MatNorm(A, NORM_1, &norms[1]));
  PetscCall(MatNorm(A, NORM_2, &norms[2]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "A Approx Matrix norms: infty=%g, norm_1=%g, norm_2=%g\n", (double)norms[0], (double)norms[1], (double)norms[2]));
  if (testnorm) {
    PetscCall(MatNorm(B, NORM_INFINITY, &norms[0]));
    PetscCall(MatNorm(B, NORM_1, &norms[1]));
    PetscCall(MatNorm(B, NORM_2, &norms[2]));
  } else {
    norms[0] = -1.;
    norms[1] = -1.;
    norms[2] = -1.;
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "B Approx Matrix norms: infty=%g, norm_1=%g, norm_2=%g\n", (double)norms[0], (double)norms[1], (double)norms[2]));
  PetscCall(MatSetOperation(A, MATOP_NORM, Anormfunc));

  /* Test MatDuplicate */
  PetscCall(MatDuplicate(B, MAT_COPY_VALUES, &D));
  PetscCall(MatSetOption(D, MAT_SYMMETRIC, symm));
  PetscCall(MatMultEqual(B, D, 10, &flg));
  if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatMult error after MatDuplicate\n"));
  if (testtrans) {
    PetscCall(MatMultTransposeEqual(B, D, 10, &flg));
    if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatMultTranspose error after MatDuplicate\n"));
  }
  PetscCall(MatDestroy(&D));

  if (testtrans) { /* MatMultTranspose for nonsymmetric matrices not implemented */
    PetscCall(VecSetRandom(Ay, NULL));
    PetscCall(VecCopy(Ay, By));
    PetscCall(MatMultTranspose(A, Ay, Ax));
    PetscCall(MatMultTranspose(B, By, Bx));
    PetscCall(VecViewFromOptions(Ax, NULL, "-multtrans_vec_view"));
    PetscCall(VecViewFromOptions(Bx, NULL, "-multtrans_vec_view"));
    PetscCall(VecNorm(Ax, NORM_INFINITY, &nX));
    PetscCall(VecAXPY(Ax, -1.0, Bx));
    PetscCall(VecViewFromOptions(Ax, NULL, "-multtrans_vec_view"));
    PetscCall(VecNorm(Ax, NORM_INFINITY, &err));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatMultTranspose err %g\n", err / nX));
    PetscCall(VecScale(Bx, -1.0));
    PetscCall(MatMultTransposeAdd(B, By, Bx, Bx));
    PetscCall(VecNorm(Bx, NORM_INFINITY, &err));
    if (err > 10. * PETSC_SMALL) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatMultTransposeAdd err %g\n", err));
  }
  PetscCall(VecDestroy(&Ax));
  PetscCall(VecDestroy(&Ay));
  PetscCall(VecDestroy(&Bx));
  PetscCall(VecDestroy(&By));

  /* Test MatMatMult */
  if (ldc) PetscCall(PetscMalloc1(nrhs * (n + ldc), &Cdata));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, n, PETSC_DECIDE, N, nrhs, Cdata, &C));
  PetscCall(MatDenseSetLDA(C, n + ldc));
  PetscCall(MatSetRandom(C, NULL));
  if (cgpu) PetscCall(MatConvert(C, MATDENSECUDA, MAT_INPLACE_MATRIX, &C));
  for (nt = 0; nt < ntrials; nt++) {
    PetscCall(MatMatMult(B, C, nt ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX, PETSC_DEFAULT, &D));
    PetscCall(MatViewFromOptions(D, NULL, "-bc_view"));
    PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)D, &flg, MATSEQDENSE, MATMPIDENSE, ""));
    if (flg) {
      PetscCall(MatCreateVecs(B, &x, &y));
      PetscCall(MatCreateVecs(D, NULL, &v));
      for (i = 0; i < nrhs; i++) {
        PetscCall(MatGetColumnVector(D, v, i));
        PetscCall(MatGetColumnVector(C, x, i));
        PetscCall(MatMult(B, x, y));
        PetscCall(VecAXPY(y, -1.0, v));
        PetscCall(VecNorm(y, NORM_INFINITY, &err));
        if (err > 10. * PETSC_SMALL) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatMat err %" PetscInt_FMT " %g\n", i, err));
      }
      PetscCall(VecDestroy(&y));
      PetscCall(VecDestroy(&x));
      PetscCall(VecDestroy(&v));
    }
  }
  PetscCall(MatDestroy(&D));

  /* Test MatTransposeMatMult */
  if (testtrans) { /* MatMultTranspose for nonsymmetric matrices not implemented */
    for (nt = 0; nt < ntrials; nt++) {
      PetscCall(MatTransposeMatMult(B, C, nt ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX, PETSC_DEFAULT, &D));
      PetscCall(MatViewFromOptions(D, NULL, "-btc_view"));
      PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)D, &flg, MATSEQDENSE, MATMPIDENSE, ""));
      if (flg) {
        PetscCall(MatCreateVecs(B, &y, &x));
        PetscCall(MatCreateVecs(D, NULL, &v));
        for (i = 0; i < nrhs; i++) {
          PetscCall(MatGetColumnVector(D, v, i));
          PetscCall(MatGetColumnVector(C, x, i));
          PetscCall(MatMultTranspose(B, x, y));
          PetscCall(VecAXPY(y, -1.0, v));
          PetscCall(VecNorm(y, NORM_INFINITY, &err));
          if (err > 10. * PETSC_SMALL) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatTransMat err %" PetscInt_FMT " %g\n", i, err));
        }
        PetscCall(VecDestroy(&y));
        PetscCall(VecDestroy(&x));
        PetscCall(VecDestroy(&v));
      }
    }
    PetscCall(MatDestroy(&D));
  }

  /* Test basis orthogonalization */
  if (testorthog) {
    PetscCall(MatDuplicate(B, MAT_COPY_VALUES, &D));
    PetscCall(MatSetOption(D, MAT_SYMMETRIC, symm));
    PetscCall(MatH2OpusOrthogonalize(D));
    PetscCall(MatMultEqual(B, D, 10, &flg));
    if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatMult error after basis ortogonalization\n"));
    PetscCall(MatDestroy(&D));
  }

  /* Test matrix compression */
  if (testcompress) {
    PetscCall(MatDuplicate(B, MAT_COPY_VALUES, &D));
    PetscCall(MatSetOption(D, MAT_SYMMETRIC, symm));
    PetscCall(MatH2OpusCompress(D, PETSC_SMALL));
    PetscCall(MatDestroy(&D));
  }

  /* Test low-rank update */
  if (testhlru) {
    Mat          U, V;
    PetscScalar *Udata = NULL, *Vdata = NULL;

    if (ldu) {
      PetscCall(PetscMalloc1(nlr * (n + ldu), &Udata));
      PetscCall(PetscMalloc1(nlr * (n + ldu + 2), &Vdata));
    }
    PetscCall(MatDuplicate(B, MAT_COPY_VALUES, &D));
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)D), n, PETSC_DECIDE, N, nlr, Udata, &U));
    PetscCall(MatDenseSetLDA(U, n + ldu));
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)D), n, PETSC_DECIDE, N, nlr, Vdata, &V));
    if (ldu) PetscCall(MatDenseSetLDA(V, n + ldu + 2));
    PetscCall(MatSetRandom(U, NULL));
    PetscCall(MatSetRandom(V, NULL));
    PetscCall(MatH2OpusLowRankUpdate(D, U, V, 0.5));
    PetscCall(MatH2OpusLowRankUpdate(D, U, V, -0.5));
    PetscCall(MatMultEqual(B, D, 10, &flg));
    if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatMult error after low-rank update\n"));
    PetscCall(MatDestroy(&D));
    PetscCall(MatDestroy(&U));
    PetscCall(PetscFree(Udata));
    PetscCall(MatDestroy(&V));
    PetscCall(PetscFree(Vdata));
  }

  /* check explicit operator */
  if (checkexpl) {
    Mat Be, Bet;

    PetscCall(MatComputeOperator(B, MATDENSE, &D));
    PetscCall(MatDuplicate(D, MAT_COPY_VALUES, &Be));
    PetscCall(MatNorm(D, NORM_FROBENIUS, &nB));
    PetscCall(MatViewFromOptions(D, NULL, "-expl_view"));
    PetscCall(MatAXPY(D, -1.0, A, SAME_NONZERO_PATTERN));
    PetscCall(MatViewFromOptions(D, NULL, "-diff_view"));
    PetscCall(MatNorm(D, NORM_FROBENIUS, &nD));
    PetscCall(MatNorm(A, NORM_FROBENIUS, &nA));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Approximation error %g (%g / %g, %g)\n", nD / nA, nD, nA, nB));
    PetscCall(MatDestroy(&D));

    if (testtrans) { /* MatMultTranspose for nonsymmetric matrices not implemented */
      PetscCall(MatTranspose(A, MAT_INPLACE_MATRIX, &A));
      PetscCall(MatComputeOperatorTranspose(B, MATDENSE, &D));
      PetscCall(MatDuplicate(D, MAT_COPY_VALUES, &Bet));
      PetscCall(MatNorm(D, NORM_FROBENIUS, &nB));
      PetscCall(MatViewFromOptions(D, NULL, "-expl_trans_view"));
      PetscCall(MatAXPY(D, -1.0, A, SAME_NONZERO_PATTERN));
      PetscCall(MatViewFromOptions(D, NULL, "-diff_trans_view"));
      PetscCall(MatNorm(D, NORM_FROBENIUS, &nD));
      PetscCall(MatNorm(A, NORM_FROBENIUS, &nA));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Approximation error transpose %g (%g / %g, %g)\n", nD / nA, nD, nA, nB));
      PetscCall(MatDestroy(&D));

      PetscCall(MatTranspose(Bet, MAT_INPLACE_MATRIX, &Bet));
      PetscCall(MatAXPY(Be, -1.0, Bet, SAME_NONZERO_PATTERN));
      PetscCall(MatViewFromOptions(Be, NULL, "-diff_expl_view"));
      PetscCall(MatNorm(Be, NORM_FROBENIUS, &nB));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Approximation error B - (B^T)^T %g\n", nB));
      PetscCall(MatDestroy(&Be));
      PetscCall(MatDestroy(&Bet));
    }
  }
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFree(Cdata));
  PetscCall(PetscFree(Adata));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: h2opus

#tests from kernel
   test:
     requires: h2opus
     nsize: 1
     suffix: 1
     args: -n {{17 33}} -kernel 1 -dim {{1 2 3}} -symm {{0 1}} -checkexpl -bgpu 0

   test:
     requires: h2opus
     nsize: 1
     suffix: 1_ld
     output_file: output/ex66_1.out
     args: -n 33 -kernel 1 -dim 1 -lda 13 -ldc 11 -ldu 17 -symm 0 -checkexpl -bgpu 0

   test:
     requires: h2opus cuda
     nsize: 1
     suffix: 1_cuda
     output_file: output/ex66_1.out
     args: -n {{17 33}} -kernel 1 -dim {{1 2 3}} -symm {{0 1}} -checkexpl -bgpu 1

   test:
     requires: h2opus cuda
     nsize: 1
     suffix: 1_cuda_ld
     output_file: output/ex66_1.out
     args: -n 33 -kernel 1 -dim 1 -lda 13 -ldc 11 -ldu 17 -symm 0 -checkexpl -bgpu 1

   test:
     requires: h2opus
     nsize: {{2 3}}
     suffix: 1_par
     args: -n 64 -symm -kernel 1 -dim 1 -ldc 12 -testlayout {{0 1}} -bgpu 0 -cgpu 0

   test:
     requires: h2opus cuda
     nsize: {{2 3}}
     suffix: 1_par_cuda
     args: -n 64 -symm -kernel 1 -dim 1 -ldc 12 -testlayout {{0 1}} -bgpu {{0 1}} -cgpu {{0 1}}
     output_file: output/ex66_1_par.out

#tests from matrix sampling (parallel or unsymmetric not supported)
   test:
     requires: h2opus
     nsize: 1
     suffix: 2
     args: -n {{17 33}} -kernel 0 -dim 2 -symm 1 -checkexpl -bgpu 0

   test:
     requires: h2opus cuda
     nsize: 1
     suffix: 2_cuda
     output_file: output/ex66_2.out
     args: -n {{17 29}} -kernel 0 -dim 2 -symm 1 -checkexpl -bgpu {{0 1}} -agpu {{0 1}}

#tests view operation
   test:
     requires: h2opus !cuda
     filter: grep -v " MPI process" | grep -v "\[" | grep -v "\]"
     nsize: {{1 2 3}}
     suffix: view
     args: -ng 64 -kernel 1 -dim 2 -symm 1 -checkexpl -B_view -mat_h2opus_leafsize 17 -mat_h2opus_normsamples 13 -mat_h2opus_indexmap_view ::ascii_matlab -mat_approximate_norm_samples 2 -mat_h2opus_normsamples 2

   test:
     requires: h2opus cuda
     filter: grep -v " MPI process" | grep -v "\[" | grep -v "\]"
     nsize: {{1 2 3}}
     suffix: view_cuda
     args: -ng 64 -kernel 1 -dim 2 -symm 1 -checkexpl -bgpu -B_view -mat_h2opus_leafsize 17 -mat_h2opus_normsamples 13 -mat_h2opus_indexmap_view ::ascii_matlab -mat_approximate_norm_samples 2 -mat_h2opus_normsamples 2

TEST*/
