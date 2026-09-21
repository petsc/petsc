static char help[] = "Creates a matrix, inserts some values, and tests MatCreateSubMatrices() and MatZeroEntries().\n\n";

#include <petscmat.h>
#include <petsc/private/petscimpl.h>

static PetscErrorCode TestSingleISStructureOnly(MPI_Comm comm)
{
  Mat             A;
  Mat            *reuse = NULL, *fresh;
  IS              rows, cols;
  PetscInt        N, rstart, rend, nreuse, nfresh;
  const PetscInt *ia, *ja, *iref, *jref;
  PetscMPIInt     size;
  PetscBool       done, equal;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  N = 2 * size;
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, 2, 2, N, N));
  PetscCall(MatSetType(A, MATMPIAIJ));
  PetscCall(MatSetOption(A, MAT_STRUCTURE_ONLY, PETSC_TRUE));
  PetscCall(MatMPIAIJSetPreallocation(A, 2, NULL, N - 2, NULL));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; ++i) {
    PetscCall(MatSetValue(A, i, i, 1.0, INSERT_VALUES));
    PetscCall(MatSetValue(A, i, (i + 1) % N, 1.0, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, N, 0, 1, &rows));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, N / 2, 0, 2, &cols));
  PetscCall(MatSetOption(A, MAT_SUBMAT_SINGLEIS, PETSC_TRUE));
  PetscCall(MatCreateSubMatrices(A, 1, &rows, &cols, MAT_INITIAL_MATRIX, &reuse));
  for (PetscInt iteration = 0; iteration < 2; ++iteration) {
    if (iteration) PetscCall(MatCreateSubMatrices(A, 1, &rows, &cols, MAT_REUSE_MATRIX, &reuse));
    PetscCall(MatSetOption(A, MAT_SUBMAT_SINGLEIS, PETSC_FALSE));
    PetscCall(MatCreateSubMatrices(A, 1, &rows, &cols, MAT_INITIAL_MATRIX, &fresh));
    PetscCall(MatGetRowIJ(reuse[0], 0, PETSC_FALSE, PETSC_FALSE, &nreuse, &ia, &ja, &done));
    PetscCheck(done, PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot access reused submatrix graph");
    PetscCall(MatGetRowIJ(fresh[0], 0, PETSC_FALSE, PETSC_FALSE, &nfresh, &iref, &jref, &done));
    PetscCheck(done && nreuse == nfresh, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Structure-only submatrix row counts differ");
    PetscCall(PetscArraycmp(ia, iref, nreuse + 1, &equal));
    PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Structure-only submatrix row offsets differ");
    PetscCall(PetscArraycmp(ja, jref, ia[nreuse], &equal));
    PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Structure-only submatrix column indices differ");
    PetscCall(MatRestoreRowIJ(fresh[0], 0, PETSC_FALSE, PETSC_FALSE, &nfresh, &iref, &jref, &done));
    PetscCall(MatRestoreRowIJ(reuse[0], 0, PETSC_FALSE, PETSC_FALSE, &nreuse, &ia, &ja, &done));
    PetscCall(MatDestroySubMatrices(1, &fresh));
  }
  PetscCall(MatDestroySubMatrices(1, &reuse));
  PetscCall(ISDestroy(&rows));
  PetscCall(ISDestroy(&cols));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestSingleISOutputGraph(MPI_Comm comm)
{
  Mat              A, difference;
  Mat             *reuse, *fresh;
  IS               rows, cols;
  PetscInt         rstart, rend, row = 0;
  PetscMPIInt      rank;
  PetscBool        equal;
  PetscReal        norm;
  PetscObjectState parentstate, newparentstate, nzstate, newnzstate;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, 6, 6, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(A, MATMPIAIJ));
  PetscCall(MatMPIAIJSetPreallocation(A, 2, NULL, 0, NULL));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; ++i) {
    PetscCall(MatSetValue(A, i, i, 10 + i, INSERT_VALUES));
    if (i + 1 < rend) PetscCall(MatSetValue(A, i, i + 1, 20 + i, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatGetNonzeroState(A, &parentstate));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, 4, rstart, 1, &rows));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, 4, rstart, 1, &cols));
  for (PetscInt mode = 0; mode < 4; ++mode) {
    PetscCall(MatSetOption(A, MAT_SUBMAT_SINGLEIS, PETSC_TRUE));
    PetscCall(MatCreateSubMatrices(A, 1, &rows, &cols, MAT_INITIAL_MATRIX, &reuse));
    // Exercise graph changes both before the first reuse and after its maps are built.
    if (mode >= 2) PetscCall(MatCreateSubMatrices(A, 1, &rows, &cols, MAT_REUSE_MATRIX, &reuse));
    PetscCall(MatSetOption(reuse[0], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    PetscCall(MatGetNonzeroState(reuse[0], &nzstate));
    // Change only rank 0 so other ranks can build or continue using their cached positions.
    if (!rank) {
      if (!(mode % 2)) PetscCall(MatZeroRows(reuse[0], 1, &row, 0.0, NULL, NULL));
      else PetscCall(MatSetValue(reuse[0], 0, 3, 0.0, INSERT_VALUES));
      PetscCall(MatAssemblyBegin(reuse[0], MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(reuse[0], MAT_FINAL_ASSEMBLY));
    }
    PetscCall(MatGetNonzeroState(reuse[0], &newnzstate));
    PetscCheck(rank ? newnzstate == nzstate : newnzstate != nzstate, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Submatrix graph must change only on rank 0");
    for (PetscInt iteration = 0; iteration < 2; ++iteration) {
      PetscCall(MatScale(A, 2.0));
      PetscCall(MatGetNonzeroState(A, &newparentstate));
      PetscCheck(newparentstate == parentstate, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Parent graph must remain unchanged");
      PetscCall(MatCreateSubMatrices(A, 1, &rows, &cols, MAT_REUSE_MATRIX, &reuse));
      PetscCall(MatSetOption(A, MAT_SUBMAT_SINGLEIS, PETSC_FALSE));
      PetscCall(MatCreateSubMatrices(A, 1, &rows, &cols, MAT_INITIAL_MATRIX, &fresh));
      if (!(mode % 2)) {
        PetscCall(MatEqual(reuse[0], fresh[0], &equal));
        PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Reused submatrix differs after removing a row, iteration %" PetscInt_FMT, iteration);
      }
      // An inserted explicit zero changes the graph but must not change the numerical result.
      PetscCall(MatDuplicate(reuse[0], MAT_COPY_VALUES, &difference));
      PetscCall(MatSetOption(difference, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
      PetscCall(MatAXPY(difference, -1.0, fresh[0], DIFFERENT_NONZERO_PATTERN));
      PetscCall(MatNorm(difference, NORM_INFINITY, &norm));
      PetscCheck(norm <= PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Reused submatrix differs after output graph mutation, mode %" PetscInt_FMT ", iteration %" PetscInt_FMT ": %g", mode, iteration, (double)norm);
      PetscCall(MatDestroy(&difference));
      PetscCall(MatDestroySubMatrices(1, &fresh));
    }
    PetscCall(MatDestroySubMatrices(1, &reuse));
  }
  PetscCall(ISDestroy(&rows));
  PetscCall(ISDestroy(&cols));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestSingleISReuse(MPI_Comm comm)
{
  Mat              A, retained, duplicate;
  Mat             *reuse, *fresh;
  IS               rows, cols;
  Vec              x, y, reference;
  PetscInt         mlocal = 6, nlocal, M, N, rstart, rend, nr, nc;
  PetscInt        *rowidx, *colidx;
  PetscMPIInt      rank, size;
  PetscBool        empty_rank = PETSC_FALSE, rectangular = PETSC_FALSE, equal;
  PetscReal        norm;
  PetscObjectState state, newstate, nzstate, newnzstate;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_empty_rank", &empty_rank, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_rectangular", &rectangular, NULL));
  if (empty_rank && size > 1 && rank == size - 1) mlocal = 0;
  nlocal = rectangular && mlocal ? 4 : mlocal;
  M      = 6 * (empty_rank && size > 1 ? size - 1 : size);
  N      = (rectangular ? 4 : 6) * (empty_rank && size > 1 ? size - 1 : size);
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, mlocal, nlocal, M, N));
  PetscCall(MatSetType(A, MATMPIAIJ));
  PetscCall(MatSetOptionsPrefix(A, "reuse_"));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatMPIAIJSetPreallocation(A, nlocal, NULL, N - nlocal, NULL));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; ++i) {
    for (PetscInt j = 0; j < N; ++j) {
      PetscScalar value = 5 * (i + 1) + 3 * (j + 1);

#if PetscDefined(USE_COMPLEX)
      value += PETSC_i * (i - 2 * j + 1);
#endif
      if ((i + j) % 3 != 1 || i == j) PetscCall(MatSetValue(A, i, j, value, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscMalloc2(M, &rowidx, N, &colidx));

  // Include fallback selections, empty graphs, remote-only rows, and a proper subset of owned rows.
  for (PetscInt scenario = 0; scenario < 9; ++scenario) {
    nr = M;
    nc = N / 2;
    for (PetscInt i = 0; i < nr; ++i) rowidx[i] = scenario == 1 ? M - 1 - i : i;
    for (PetscInt i = 0; i < nc; ++i) colidx[i] = 2 * (scenario == 2 ? nc - 1 - i : i);
    if (scenario == 3) {
      nr = M / 2;
      nc = N;
      for (PetscInt i = 0; i < nr; ++i) rowidx[i] = 2 * i;
      for (PetscInt i = 0; i < nc; ++i) colidx[i] = i;
    }
    if (scenario == 4 && !rank) nr = nc = 0;
    if (scenario == 5) {
      nr = 0;
      for (PetscInt i = 0; i < M; ++i) {
        if (i < rstart || i >= rend) rowidx[nr++] = i;
      }
    }
    if (scenario == 6) {
      nr = M / 2;
      for (PetscInt i = 0; i < nr; ++i) rowidx[i] = 2 * i;
    }
    if (scenario == 7) nc = 0;
    if (scenario == 8) {
      nr = nc   = 1;
      rowidx[0] = 0;
      colidx[0] = 1; // This entry is absent from the parent graph.
    }
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nr, rowidx, PETSC_COPY_VALUES, &rows));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nc, colidx, PETSC_COPY_VALUES, &cols));
    PetscCall(MatSetOption(A, MAT_SUBMAT_SINGLEIS, PETSC_TRUE));
    PetscCall(MatCreateSubMatrices(A, 1, &rows, &cols, MAT_INITIAL_MATRIX, &reuse));
    PetscCall(MatCreateVecs(reuse[0], &x, &y));
    PetscCall(VecDuplicate(y, &reference));
    PetscCall(VecSet(x, 1.0));
    for (PetscInt iteration = 0; iteration < 3; ++iteration) {
      // Exercise cached/device values before changing the parent.
      PetscCall(MatMult(reuse[0], x, y));
      PetscCall(PetscObjectStateGet((PetscObject)reuse[0], &state));
      PetscCall(MatGetNonzeroState(reuse[0], &nzstate));
      PetscCall(MatScale(A, -0.5));
      if (!rectangular) PetscCall(MatShift(A, iteration + 1));
      PetscCall(MatCreateSubMatrices(A, 1, &rows, &cols, MAT_REUSE_MATRIX, &reuse));
      PetscCall(PetscObjectStateGet((PetscObject)reuse[0], &newstate));
      PetscCall(MatGetNonzeroState(reuse[0], &newnzstate));
      PetscCheck(newstate > state && newnzstate == nzstate, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Reusing a submatrix must update its values without changing its graph");
      // A fresh extraction uses the general path, independently of the cached SingleIS maps.
      PetscCall(MatSetOption(A, MAT_SUBMAT_SINGLEIS, PETSC_FALSE));
      PetscCall(MatCreateSubMatrices(A, 1, &rows, &cols, MAT_INITIAL_MATRIX, &fresh));
      PetscCall(MatEqual(reuse[0], fresh[0], &equal));
      PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Reused submatrix differs from fresh extraction in scenario %" PetscInt_FMT ", iteration %" PetscInt_FMT, scenario, iteration);
      PetscCall(MatMult(reuse[0], x, y));
      PetscCall(MatMult(fresh[0], x, reference));
      PetscCall(VecAXPY(y, -1.0, reference));
      PetscCall(VecNorm(y, NORM_INFINITY, &norm));
      PetscCheck(norm <= PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Reused submatrix product differs from fresh extraction: %g", (double)norm);
      PetscCall(MatDestroySubMatrices(1, &fresh));
    }
    retained = reuse[0];
    PetscCall(PetscObjectReference((PetscObject)retained));
    PetscCall(MatDuplicate(retained, MAT_COPY_VALUES, &duplicate));
    PetscCall(MatDestroySubMatrices(1, &reuse));
    PetscCall(MatEqual(retained, duplicate, &equal));
    PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Retained submatrix values changed after destroying its containing array");
    PetscCall(MatMult(retained, x, y));
    PetscCall(MatDestroy(&duplicate));
    PetscCall(MatDestroy(&retained));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&y));
    PetscCall(VecDestroy(&reference));
    PetscCall(ISDestroy(&rows));
    PetscCall(ISDestroy(&cols));
  }
  PetscCall(PetscFree2(rowidx, colidx));
  PetscCall(MatDestroy(&A));
  PetscCall(TestSingleISStructureOnly(comm));
  PetscCall(TestSingleISOutputGraph(comm));
  PetscCall(PetscPrintf(comm, "Submatrix reuse tests passed\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat         mat, submat, submat1;
  Mat        *submatrices;
  PetscInt    m = 10, n = 10, i = 4, tmp, rstart, rend;
  IS          irow, icol;
  PetscScalar value = 1.0;
  PetscViewer sviewer;
  PetscBool   allA = PETSC_FALSE, test_reuse = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_reuse", &test_reuse, NULL));
  if (test_reuse) PetscCall(TestSingleISReuse(PETSC_COMM_WORLD));
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_COMMON));
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_SELF, PETSC_VIEWER_ASCII_COMMON));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &mat));
  PetscCall(MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, m, n));
  PetscCall(MatSetFromOptions(mat));
  PetscCall(MatSetUp(mat));
  PetscCall(MatGetOwnershipRange(mat, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    value = (PetscReal)i + 1;
    tmp   = i % 5;
    PetscCall(MatSetValues(mat, 1, &tmp, 1, &i, &value, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Original matrix\n"));
  PetscCall(MatView(mat, PETSC_VIEWER_STDOUT_WORLD));

  /* Test MatCreateSubMatrix_XXX_All(), i.e., submatrix = A */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_all", &allA, NULL));
  if (allA) {
    PetscCall(ISCreateStride(PETSC_COMM_SELF, m, 0, 1, &irow));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, n, 0, 1, &icol));
    PetscCall(MatCreateSubMatrices(mat, 1, &irow, &icol, MAT_INITIAL_MATRIX, &submatrices));
    PetscCall(MatCreateSubMatrices(mat, 1, &irow, &icol, MAT_REUSE_MATRIX, &submatrices));
    submat = *submatrices;

    /* sviewer will cause the submatrices (one per processor) to be printed in the correct order */
    PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "\nSubmatrices with all\n"));
    PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "--------------------\n"));
    PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &sviewer));
    PetscCall(MatView(submat, sviewer));
    PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &sviewer));

    PetscCall(ISDestroy(&irow));
    PetscCall(ISDestroy(&icol));

    /* test getting a reference on a submat */
    PetscCall(PetscObjectReference((PetscObject)submat));
    PetscCall(MatDestroySubMatrices(1, &submatrices));
    PetscCall(MatDestroy(&submat));
  }

  /* Form submatrix with rows 2-4 and columns 4-8 */
  PetscCall(ISCreateStride(PETSC_COMM_SELF, 3, 2, 1, &irow));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, 5, 4, 1, &icol));
  PetscCall(MatCreateSubMatrices(mat, 1, &irow, &icol, MAT_INITIAL_MATRIX, &submatrices));
  submat = *submatrices;

  /* Test reuse submatrices */
  PetscCall(MatCreateSubMatrices(mat, 1, &irow, &icol, MAT_REUSE_MATRIX, &submatrices));

  /* sviewer will cause the submatrices (one per processor) to be printed in the correct order */
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "\nSubmatrices\n"));
  PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &sviewer));
  PetscCall(MatView(submat, sviewer));
  PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &sviewer));
  PetscCall(PetscObjectReference((PetscObject)submat));
  PetscCall(MatDestroySubMatrices(1, &submatrices));
  PetscCall(MatDestroy(&submat));

  /* Form submatrix with rows 2-4 and all columns */
  PetscCall(ISDestroy(&icol));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, 10, 0, 1, &icol));
  PetscCall(MatCreateSubMatrices(mat, 1, &irow, &icol, MAT_INITIAL_MATRIX, &submatrices));
  PetscCall(MatCreateSubMatrices(mat, 1, &irow, &icol, MAT_REUSE_MATRIX, &submatrices));
  submat = *submatrices;

  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "\nSubmatrices with allcolumns\n"));
  PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &sviewer));
  PetscCall(MatView(submat, sviewer));
  PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &sviewer));

  /* Test MatDuplicate */
  PetscCall(MatDuplicate(submat, MAT_COPY_VALUES, &submat1));
  PetscCall(MatDestroy(&submat1));

  /* Zero the original matrix */
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Original zeroed matrix\n"));
  PetscCall(MatZeroEntries(mat));
  PetscCall(MatView(mat, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(ISDestroy(&irow));
  PetscCall(ISDestroy(&icol));
  PetscCall(PetscObjectReference((PetscObject)submat));
  PetscCall(MatDestroySubMatrices(1, &submatrices));
  PetscCall(MatDestroy(&submat));
  PetscCall(MatDestroy(&mat));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -mat_type aij

   test:
      suffix: 2
      args: -mat_type dense

   test:
      suffix: 3
      nsize: 3
      args: -mat_type aij

   test:
      suffix: 4
      nsize: 3
      args: -mat_type dense

   test:
      suffix: 5
      nsize: 3
      args: -mat_type aij -test_all

   test:
      suffix: singleis_reuse
      nsize: {{1 2 3}}
      args: -mat_type aij -test_reuse -test_empty_rank {{0 1}}
      filter: grep "^Submatrix reuse"
      output_file: output/ex4_singleis_reuse.out

   test:
      suffix: singleis_reuse_rectangular
      nsize: 3
      args: -mat_type aij -test_reuse -test_rectangular -test_empty_rank {{0 1}}
      filter: grep "^Submatrix reuse"
      output_file: output/ex4_singleis_reuse.out

   test:
      suffix: singleis_reuse_cuda
      requires: cuda
      nsize: 3
      args: -mat_type aij -test_reuse -test_empty_rank {{0 1}} -reuse_mat_type mpiaijcusparse
      filter: grep "^Submatrix reuse"
      output_file: output/ex4_singleis_reuse.out

   test:
      suffix: singleis_reuse_kokkos
      requires: kokkos_kernels
      nsize: 3
      args: -mat_type aij -test_reuse -test_empty_rank {{0 1}} -reuse_mat_type mpiaijkokkos
      filter: grep "^Submatrix reuse"
      output_file: output/ex4_singleis_reuse.out

TEST*/
