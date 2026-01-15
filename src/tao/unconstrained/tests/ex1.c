static char help[] = "test least-squares problem created from a mapped taoterm quadratic";

#include <petsctao.h>

int main(int argc, char **argv)
{
  MPI_Comm    comm;
  Mat         A;       // data matrix
  Mat         W;       // weight matrix
  Vec         w;       // observation vector
  Vec         b;       // observation vector
  PetscInt    m = 100; // data size
  PetscInt    n = 20;  // model size
  TaoTerm     data_term;
  PetscRandom rand;
  Tao         tao;
  PetscInt    i, j;
  PetscReal   val, density = 0.3;
  PetscBool   test_quad_mat = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscOptionsBegin(comm, "", help, "none");
  PetscCall(PetscOptionsBoundedInt("-m", "data size", "", m, &m, NULL, 0));
  PetscCall(PetscOptionsBoundedInt("-n", "model size", "", n, &n, NULL, 0));
  PetscCall(PetscOptionsBool("-test_quad_mat", "Test if quadratic term matrix matches W matrix", "", test_quad_mat, &test_quad_mat, NULL));
  PetscOptionsEnd();

  PetscCall(TaoCreate(comm, &tao));

  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(PetscRandomSetInterval(rand, -1.0, 1.0));
  PetscCall(PetscRandomSetFromOptions(rand));

  // create the model data, A, W and b
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, n));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      PetscCall(PetscRandomGetValueReal(rand, &val));
      // Optionally make it sparse: only insert some entries
      if (val < density) PetscCall(MatSetValue(A, i, j, val, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(VecCreateMPI(comm, PETSC_DECIDE, m, &b));
  PetscCall(VecSetRandom(b, rand));
  PetscCall(VecDuplicate(b, &w));
  PetscCall(VecSetRandom(w, rand));
  PetscCall(VecAbs(w));
  PetscCall(VecShift(w, 1.0));
  PetscCall(MatCreateDiagonal(w, &W));
  PetscCall(VecDestroy(&w));

  // the model term,  (1/2) || Ax - b ||_W^2
  PetscCall(TaoTermCreateQuadratic(W, &data_term));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)data_term, "data_"));
  PetscCall(TaoAddTerm(tao, "data_", 3.0, data_term, b, A));
  PetscCall(TaoTermDestroy(&data_term));

  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));

  if (test_quad_mat) {
    PetscReal   scale;
    TaoTerm     term;
    Vec         params;
    Mat         map;
    Mat         quad_mat;
    TaoTermType term_type;
    PetscBool   is_quad, mat_equal;

    PetscCall(TaoGetTerm(tao, &scale, &term, &params, &map));
    PetscCall(TaoTermGetType(term, &term_type));
    PetscCall(PetscStrcmp(term_type, TAOTERMQUADRATIC, &is_quad));
    PetscCheck(is_quad, comm, PETSC_ERR_ARG_WRONG, "Term from TaoGetTerm is not a quadratic term");

    PetscCall(TaoTermQuadraticGetMat(term, &quad_mat));
    PetscCheck(quad_mat != NULL, comm, PETSC_ERR_ARG_NULL, "Quadratic term matrix is NULL");

    PetscCall(MatEqual(W, quad_mat, &mat_equal));
    PetscCheck(mat_equal, comm, PETSC_ERR_PLIB, "Quadratic term matrix does not match W matrix");
    PetscCall(PetscPrintf(comm, "Test passed: Quadratic term matrix matches W matrix\n"));
  }

  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&W));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(TaoDestroy(&tao));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex !single !quad !defined(PETSC_USE_64BIT_INDICES) !__float128

  test:
    suffix: 0
    args: -tao_monitor_short -tao_view -tao_type nls

  test:
    suffix: 1
    args: -tao_view ::ascii_info_detail -tao_type nls

  test:
    suffix: test_quad_mat
    args: -test_quad_mat 1

TEST*/
