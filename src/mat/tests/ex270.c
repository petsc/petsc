#include <petscsys.h>
#include <petscmat.h>
#include <petscoptions.h>

static const char help[] = "Test MatGetValue/Row for hypre matrix on device\n";

static PetscErrorCode CheckBandedMatrix(MPI_Comm comm, PetscInt N, PetscInt bw, const char fill[])
{
  Mat         A;
  PetscInt    rstart, rend;
  PetscBool   iscoo, isconvert, ok = PETSC_TRUE;
  PetscScalar expected;

  PetscFunctionBeginUser;
  PetscCall(PetscStrcmp(fill, "coo", &iscoo));
  PetscCall(PetscStrcmp(fill, "convert", &isconvert));

  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N));
  PetscCall(MatSetType(A, isconvert ? MATAIJ : MATHYPRE));
  PetscCall(MatSeqAIJSetPreallocation(A, 2 * bw + 1, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 2 * bw + 1, NULL, 2 * bw + 1, NULL));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));

  if (iscoo) {
    PetscCount   ncoo = 0;
    PetscInt    *coo_i, *coo_j;
    PetscInt     nmax = (rend - rstart) * (2 * bw + 1); /* an upper bound; rows at the ends hold fewer */
    PetscScalar *coo_v;

    PetscCall(PetscMalloc3(nmax, &coo_i, nmax, &coo_j, nmax, &coo_v));
    for (PetscInt i = rstart; i < rend; i++) {
      for (PetscInt j = PetscMax(0, i - bw); j <= PetscMin(N - 1, i + bw); j++) {
        coo_i[ncoo] = i;
        coo_j[ncoo] = j;
        coo_v[ncoo] = (PetscScalar)(100 * (i + 1) + j + 1);
        ncoo++;
      }
    }
    PetscCall(MatSetPreallocationCOO(A, ncoo, coo_i, coo_j));
    PetscCall(MatSetValuesCOO(A, coo_v, INSERT_VALUES));
    PetscCall(PetscFree3(coo_i, coo_j, coo_v));
  } else {
    for (PetscInt i = rstart; i < rend; i++) {
      for (PetscInt j = PetscMax(0, i - bw); j <= PetscMin(N - 1, i + bw); j++) {
        expected = (PetscScalar)(100 * (i + 1) + j + 1);
        PetscCall(MatSetValues(A, 1, &i, 1, &j, &expected, INSERT_VALUES));
      }
    }
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }

  if (isconvert) {
    Mat H;

    PetscCall(MatConvert(A, MATHYPRE, MAT_INITIAL_MATRIX, &H));
    PetscCall(MatDestroy(&A));
    A = H;
    /* the conversion leaves the matrix on the host; the reordering only matters once it moves */
    PetscCall(MatBindToCPU(A, PETSC_FALSE));
  }

  for (PetscInt i = rstart; i < rend && ok; i++) {
    const PetscInt    *cols;
    const PetscScalar *rowvals;
    PetscInt           ncols;

    /* MatGetValues() over the whole row range, so that structural zeros are covered too */
    for (PetscInt j = 0; j < N; j++) {
      PetscScalar got;

      expected = (j >= i - bw && j <= i + bw) ? (PetscScalar)(100 * (i + 1) + j + 1) : 0.0;
      PetscCall(MatGetValues(A, 1, &i, 1, &j, &got));
      if (PetscAbsScalar(got - expected) > PETSC_SMALL) ok = PETSC_FALSE;
    }

    PetscCall(MatGetRow(A, i, &ncols, &cols, &rowvals));
    for (PetscInt k = 0; k < ncols; k++) {
      expected = (PetscScalar)(100 * (i + 1) + cols[k] + 1);
      if (PetscAbsScalar(rowvals[k] - expected) > PETSC_SMALL) ok = PETSC_FALSE;
    }
    PetscCall(MatRestoreRow(A, i, &ncols, &cols, &rowvals));
  }

  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_C_BOOL, MPI_LAND, comm));
  PetscCall(PetscPrintf(comm, "Banded matrix check (%s fill): %s\n", fill, ok ? "OK" : "FAILED"));

  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscInt  n = 10;   // elements
  PetscReal L = 10.0; // domain length

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-L", &L, NULL));

  const PetscInt  N = n + 1; // nodes
  const PetscReal h = L / (PetscReal)n;

  Mat M;
  PetscCall(MatCreate(PETSC_COMM_WORLD, &M));
  PetscCall(MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE, N, N));
  PetscCall(MatSetType(M, MATAIJ));
  PetscCall(MatSetFromOptions(M));
  // Tridiagonal pattern: up to 3 nnz/row (ends have 2). Use same for on/off diag for simplicity.
  PetscCall(MatSetUp(M));
  PetscCall(MatMPIAIJSetPreallocation(M, 3, NULL, 3, NULL));
  PetscCall(MatSeqAIJSetPreallocation(M, 3, NULL));

  // Ownership range for rows (nodes)
  PetscInt rstart, rend;
  PetscCall(MatGetOwnershipRange(M, &rstart, &rend));

  // Element matrix (scaled)
  const PetscScalar s     = (PetscScalar)(h / 6.0);
  const PetscScalar ae[4] = {2 * s, 1 * s, 1 * s, 2 * s};

  // Assemble by looping over elements that start at locally-owned row i
  PetscInt idx[2];
  for (PetscInt i = PetscMax(rstart, 0); i < PetscMin(rend, n); ++i) {
    idx[0] = i;
    idx[1] = i + 1;
    PetscCall(MatSetValues(M, 2, idx, 2, idx, ae, ADD_VALUES));
  }

  PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));

  // --- Verification of MatGetRow ------------------------------------------
  {
    const PetscReal tol = 100 * PETSC_MACHINE_EPSILON; // tight but safe
    PetscBool       ok  = PETSC_TRUE;

    for (PetscInt i = rstart; i < rend; ++i) {
      const PetscInt    *cols;
      const PetscScalar *getrowvals;
      PetscScalar       *getvals;
      PetscInt           ncols = 0, expected_ncols = 0;
      PetscInt           expected_cols[3];
      PetscScalar        expected_vals[3];

      // Build expected data
      if (i > 0) {
        expected_cols[expected_ncols] = i - 1;
        expected_vals[expected_ncols] = s;
        ++expected_ncols;
      }
      expected_cols[expected_ncols] = i;
      expected_vals[expected_ncols] = (i == 0 || i == N - 1) ? (2 * s) : (4 * s); // diag: h/3 at ends, 2h/3 interior
      ++expected_ncols;
      if (i + 1 < N) {
        expected_cols[expected_ncols] = i + 1;
        expected_vals[expected_ncols] = s;
        ++expected_ncols;
      }

      PetscCall(PetscMalloc1(expected_ncols, &getvals));
      PetscCall(MatGetRow(M, i, &ncols, &cols, &getrowvals));
      PetscCall(MatGetValues(M, 1, &i, expected_ncols, (PetscInt *)expected_cols, getvals));

      // Compare counts
      if (ncols != expected_ncols) {
        ok = PETSC_FALSE;
        goto rowdone;
      }

      // Compare values (match by column)
      for (PetscInt k = 0; k < ncols; ++k) {
        PetscInt expected_k = -1;
        /* Matrix is small. Just do a linear search */
        for (PetscInt l = 0; l < expected_ncols; ++l) {
          if (expected_cols[l] == cols[k]) {
            expected_k = l;
            break;
          }
        }
        if (expected_k < 0) {
          ok = PETSC_FALSE;
          goto rowdone;
        }
        if ((PetscAbsScalar(getrowvals[k] - expected_vals[expected_k]) > tol) || (PetscAbsScalar(getvals[expected_k] - expected_vals[expected_k]) > tol)) {
          ok = PETSC_FALSE;
          goto rowdone;
        }
      }

    rowdone:
      PetscCall(MatRestoreRow(M, i, &ncols, &cols, &getrowvals));
      PetscCall(PetscFree(getvals));
      if (!ok) break;
    }

    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_C_BOOL, MPI_LAND, PETSC_COMM_WORLD));
    if (ok) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Mass matrix check: OK\n"));
    else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Mass matrix check: FAILED\n"));
  }
  // --------------------------------------------------------------------------

  PetscCall(MatDestroy(&M));

  {
    PetscInt  bw = 3;
    PetscBool flg;
    char      fill[16];

    PetscCall(PetscOptionsGetInt(NULL, NULL, "-bw", &bw, NULL));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-fill", fill, sizeof(fill), &flg));
    if (!flg) PetscCall(PetscStrncpy(fill, "setvalues", sizeof(fill)));
    PetscCall(CheckBandedMatrix(PETSC_COMM_WORLD, N, bw, fill));
  }

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      requires: hypre
      suffix: 1
      args: -mat_type hypre

   test:
      requires: hypre
      suffix: 2
      args: -mat_type hypre
      nsize: 4

   test:
      requires: hypre
      suffix: coo
      output_file: output/ex270_coo.out
      args: -mat_type hypre -fill coo
      nsize: {{1 4}}

   test:
      requires: hypre
      suffix: convert
      output_file: output/ex270_convert.out
      args: -mat_type hypre -fill convert
      nsize: {{1 4}}

TEST*/
