static char help[] = "Mini-app to benchmark matrix--matrix multiplication\n\n";

/*
  See the paper below for more information

   "KSPHPDDM and PCHPDDM: Extending PETSc with Robust Overlapping Schwarz Preconditioners and Advanced Krylov Methods",
   P. Jolivet, J. E. Roman, and S. Zampini (2020).
*/

#include <petsc.h>

#if defined(PETSC_HAVE_MKL)
#include <mkl.h>
#define PetscStackCallMKLSparse(func, args) do {               \
    sparse_status_t __ierr;                                    \
    PetscStackPush(#func);                                     \
    __ierr = func args;                                        \
    PetscStackPop;                                             \
    if (__ierr != SPARSE_STATUS_SUCCESS) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in %s(): error code %d", #func, (int)__ierr); \
  } while (0)
#else
#define PetscStackCallMKLSparse(func, args) do {               \
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No MKL support"); \
  } while (0)
#endif

int main(int argc, char** argv) {
  Mat            A, C, D, E;
  PetscInt       nbs = 10, ntype = 10, nN = 8, m, M, trial = 5;
  PetscViewer    viewer;
  PetscInt       bs[10], N[8];
  char           *type[10];
  PetscMPIInt    size;
  PetscBool      flg, cuda, maij = PETSC_FALSE, check = PETSC_FALSE, trans = PETSC_FALSE, convert = PETSC_FALSE, mkl;
  char           file[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only");
  ierr = PetscOptionsGetString(NULL, NULL, "-f", file, PETSC_MAX_PATH_LEN, &flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Must indicate binary file with the -f option");
  ierr = PetscOptionsGetInt(NULL, NULL, "-trial", &trial, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetIntArray(NULL, NULL, "-bs", bs, &nbs, &flg);CHKERRQ(ierr);
  if (!flg) {
    nbs = 1;
    bs[0] = 1;
  }
  ierr = PetscOptionsGetIntArray(NULL, NULL, "-N", N, &nN, &flg);CHKERRQ(ierr);
  if (!flg) {
    nN = 8;
    N[0] = 1;  N[1] = 2;  N[2] = 4;  N[3] = 8;
    N[4] = 16; N[5] = 32; N[6] = 64; N[7] = 128;
  }
  ierr = PetscOptionsGetStringArray(NULL, NULL, "-type", type, &ntype, &flg);CHKERRQ(ierr);
  if (!flg) {
    ntype = 1;
    ierr = PetscStrallocpy(MATSEQAIJ, &type[0]);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetBool(NULL, NULL, "-check", &check, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-trans", &trans, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-convert_aij", &convert, NULL);CHKERRQ(ierr);
  for (PetscInt j = 0; j < nbs; ++j) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &viewer);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatLoad(A, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompareAny((PetscObject)A, &flg, MATSEQAIJ, MATMPIAIJ, "");CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Must indicate a MatAIJ input matrix");
    ierr = MatGetSize(A, &m, &M);CHKERRQ(ierr);
    if (m == M) {
      Mat oA;
      ierr = MatTranspose(A, MAT_INITIAL_MATRIX, &oA);CHKERRQ(ierr);
      ierr = MatAXPY(A, 1.0, oA, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatDestroy(&oA);CHKERRQ(ierr);
    }
    ierr = MatGetLocalSize(A, &m, NULL);CHKERRQ(ierr);
    ierr = MatGetSize(A, &M, NULL);CHKERRQ(ierr);
    if (bs[j] > 1) {
      Mat               T, Tt, B;
      const PetscScalar *ptr;
      PetscScalar       *val, *Aa;
      const PetscInt    *Ai, *Aj;
      PetscInt          An, i, k;
      PetscBool         done;

      ierr = MatCreateDense(PETSC_COMM_SELF, bs[j], bs[j], bs[j], bs[j], NULL, &T);CHKERRQ(ierr);
      ierr = MatSetRandom(T, NULL);CHKERRQ(ierr);
      ierr = MatTranspose(T, MAT_INITIAL_MATRIX, &Tt);CHKERRQ(ierr);
      ierr = MatAXPY(T, 1.0, Tt, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatDestroy(&Tt);CHKERRQ(ierr);
      ierr = MatDenseGetArrayRead(T, &ptr);CHKERRQ(ierr);
      ierr = MatGetRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &An, &Ai, &Aj, &done);CHKERRQ(ierr);
      if (!done || An != m) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Inconsistent sizes");
      ierr = MatSeqAIJGetArray(A, &Aa);CHKERRQ(ierr);
      ierr = MatCreate(PETSC_COMM_WORLD, &B);CHKERRQ(ierr);
      ierr = MatSetType(B, MATSEQBAIJ);CHKERRQ(ierr);
      ierr = MatSetSizes(B, bs[j] * An, bs[j] * An, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr);
      ierr = PetscMalloc1(Ai[An] * bs[j] * bs[j],&val);CHKERRQ(ierr);
      for (i = 0; i < Ai[An]; ++i)
        for (k = 0; k < bs[j] * bs[j]; ++k)
          val[i * bs[j] * bs[j] + k] = Aa[i] * ptr[k];
      ierr = MatSetOption(B, MAT_ROW_ORIENTED, PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatSeqBAIJSetPreallocationCSR(B, bs[j], Ai, Aj, val);CHKERRQ(ierr);
      ierr = PetscFree(val);CHKERRQ(ierr);
      ierr = MatSeqAIJRestoreArray(A, &Aa);CHKERRQ(ierr);
      ierr = MatRestoreRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &An, &Ai, &Aj, &done);CHKERRQ(ierr);
      ierr = MatDenseRestoreArrayRead(T, &ptr);CHKERRQ(ierr);
      ierr = MatDestroy(&T);CHKERRQ(ierr);
      ierr = MatDestroy(&A);CHKERRQ(ierr);
      A    = B;
    }
    /* reconvert back to SeqAIJ before converting to the desired type later */
    if (!convert) E = A;
    ierr = MatConvert(A, MATSEQAIJ, convert ? MAT_INITIAL_MATRIX : MAT_INPLACE_MATRIX, &E);CHKERRQ(ierr);
    ierr = MatSetOption(E, MAT_SYMMETRIC, PETSC_TRUE);CHKERRQ(ierr);
    for (PetscInt i = 0; i < ntype; ++i) {
      char        *tmp;
      PetscInt    *ia_ptr, *ja_ptr, k;
      PetscScalar *a_ptr;
#if defined(PETSC_HAVE_MKL)
      struct matrix_descr descr;
      sparse_matrix_t     spr;
      descr.type = SPARSE_MATRIX_TYPE_GENERAL;
      descr.diag = SPARSE_DIAG_NON_UNIT;
#endif
      if (convert) {
        ierr = MatDestroy(&A);CHKERRQ(ierr);
      }
      ierr = PetscStrstr(type[i], "mkl", &tmp);CHKERRQ(ierr);
      if (tmp) {
        size_t mlen, tlen;
        char base[256];

        mkl  = PETSC_TRUE;
        ierr = PetscStrlen(tmp, &mlen);CHKERRQ(ierr);
        ierr = PetscStrlen(type[i], &tlen);CHKERRQ(ierr);
        ierr = PetscStrncpy(base, type[i], tlen-mlen + 1);CHKERRQ(ierr);
        ierr = MatConvert(E, base, convert ? MAT_INITIAL_MATRIX : MAT_INPLACE_MATRIX, &A);CHKERRQ(ierr);
      } else {
        mkl  = PETSC_FALSE;
        ierr = PetscStrstr(type[i], "maij", &tmp);CHKERRQ(ierr);
        if (!tmp) {
          ierr = MatConvert(E, type[i], convert ? MAT_INITIAL_MATRIX : MAT_INPLACE_MATRIX, &A);CHKERRQ(ierr);
        } else {
          ierr = MatConvert(E, MATAIJ, convert ? MAT_INITIAL_MATRIX : MAT_INPLACE_MATRIX, &A);CHKERRQ(ierr);
          maij = PETSC_TRUE;
        }
      }
      ierr = PetscObjectTypeCompareAny((PetscObject)A, &cuda, MATSEQAIJCUSPARSE, MATMPIAIJCUSPARSE, "");CHKERRQ(ierr);
      if (mkl) {
        const PetscInt *Ai, *Aj;
        PetscInt       An;
        PetscBool      done;

        ierr = PetscObjectTypeCompareAny((PetscObject)A, &flg, MATSEQAIJ, MATSEQBAIJ, MATSEQSBAIJ, "");CHKERRQ(ierr);
        if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Not implemented");
        ierr = PetscObjectTypeCompare((PetscObject)A, MATSEQAIJ, &flg);CHKERRQ(ierr);
        ierr = MatGetRowIJ(A, 0, PETSC_FALSE, flg ? PETSC_FALSE : PETSC_TRUE, &An, &Ai, &Aj, &done);CHKERRQ(ierr);
        if (!done) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Inconsistent sizes");
        ierr = PetscMalloc1(An + 1,&ia_ptr);CHKERRQ(ierr);
        ierr = PetscMalloc1(Ai[An],&ja_ptr);CHKERRQ(ierr);
        if (flg) { /* SeqAIJ */
          for (k = 0; k < An + 1; ++k) ia_ptr[k] = Ai[k];
          for (k = 0; k < Ai[An]; ++k) ja_ptr[k] = Aj[k];
          ierr = MatSeqAIJGetArray(A, &a_ptr);CHKERRQ(ierr);
          PetscStackCallMKLSparse(mkl_sparse_d_create_csr, (&spr, SPARSE_INDEX_BASE_ZERO, An, An, ia_ptr, ia_ptr + 1, ja_ptr, a_ptr));
        } else {
          ierr = PetscObjectTypeCompare((PetscObject)A, MATSEQBAIJ, &flg);CHKERRQ(ierr);
          if (flg) {
            for (k = 0; k < An + 1; ++k) ia_ptr[k] = Ai[k] + 1; /* Fortran indexing to maximize cases covered by _mm routines */
            for (k = 0; k < Ai[An]; ++k) ja_ptr[k] = Aj[k] + 1; /* Fortran indexing to maximize cases covered by _mm routines */
            ierr = MatSeqBAIJGetArray(A, &a_ptr);CHKERRQ(ierr);
            PetscStackCallMKLSparse(mkl_sparse_d_create_bsr, (&spr, SPARSE_INDEX_BASE_ONE, SPARSE_LAYOUT_COLUMN_MAJOR, An, An, bs[j], ia_ptr, ia_ptr + 1, ja_ptr, a_ptr));
          } else {
            ierr = PetscObjectTypeCompare((PetscObject)A, MATSEQSBAIJ, &flg);CHKERRQ(ierr);
            if (flg) {
              for (k = 0; k < An + 1; ++k) ia_ptr[k] = Ai[k] + 1; /* Fortran indexing to maximize cases covered by _mm routines */
              for (k = 0; k < Ai[An]; ++k) ja_ptr[k] = Aj[k] + 1; /* Fortran indexing to maximize cases covered by _mm routines */
              ierr = MatSeqSBAIJGetArray(A, &a_ptr);CHKERRQ(ierr);
              PetscStackCallMKLSparse(mkl_sparse_d_create_bsr, (&spr, SPARSE_INDEX_BASE_ONE, SPARSE_LAYOUT_COLUMN_MAJOR, An, An, bs[j], ia_ptr, ia_ptr + 1, ja_ptr, a_ptr));
#if defined(PETSC_HAVE_MKL)
              descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
              descr.mode = SPARSE_FILL_MODE_UPPER;
              descr.diag = SPARSE_DIAG_NON_UNIT;
#endif
            }
          }
        }
        ierr = PetscObjectTypeCompare((PetscObject)A, MATSEQAIJ, &flg);CHKERRQ(ierr);
        ierr = MatRestoreRowIJ(A, 0, PETSC_FALSE, flg ? PETSC_FALSE : PETSC_TRUE, &An, &Ai, &Aj, &done);CHKERRQ(ierr);
      }

      ierr = MatViewFromOptions(A, NULL, "-A_view");CHKERRQ(ierr);

      for (k = 0; k < nN; ++k) {
        MatType       Atype, Ctype;
        PetscInt      AM, AN, CM, CN, t;
        PetscLogStage stage, tstage;
        char          stage_s[256];

        ierr = MatCreateDense(PETSC_COMM_WORLD, bs[j] * m, PETSC_DECIDE, bs[j] * M, N[k], NULL, &C);CHKERRQ(ierr);
        ierr = MatCreateDense(PETSC_COMM_WORLD, bs[j] * m, PETSC_DECIDE, bs[j] * M, N[k], NULL, &D);CHKERRQ(ierr);
        ierr = MatSetRandom(C, NULL);CHKERRQ(ierr);
        if (cuda) { /* convert to GPU if needed */
          ierr = MatConvert(C, MATDENSECUDA, MAT_INPLACE_MATRIX, &C);CHKERRQ(ierr);
          ierr = MatConvert(D, MATDENSECUDA, MAT_INPLACE_MATRIX, &D);CHKERRQ(ierr);
        }
        if (mkl) {
          if (N[k] > 1) PetscStackCallMKLSparse(mkl_sparse_set_mm_hint, (spr, SPARSE_OPERATION_NON_TRANSPOSE, descr, SPARSE_LAYOUT_COLUMN_MAJOR, N[k], 1 + trial));
          else          PetscStackCallMKLSparse(mkl_sparse_set_mv_hint, (spr, SPARSE_OPERATION_NON_TRANSPOSE, descr, 1 + trial));
          PetscStackCallMKLSparse(mkl_sparse_set_memory_hint, (spr, SPARSE_MEMORY_AGGRESSIVE));
          PetscStackCallMKLSparse(mkl_sparse_optimize, (spr));
        }
        ierr = MatGetType(A, &Atype);CHKERRQ(ierr);
        ierr = MatGetType(C, &Ctype);CHKERRQ(ierr);
        ierr = MatGetSize(A, &AM, &AN);CHKERRQ(ierr);
        ierr = MatGetSize(C, &CM, &CN);CHKERRQ(ierr);

        if (!maij || N[k] > 1) {
          ierr = PetscSNPrintf(stage_s, sizeof(stage_s), "type_%s-bs_%D-N_%02d", type[i], bs[j], (int)N[k]);CHKERRQ(ierr);
          ierr = PetscLogStageRegister(stage_s, &stage);CHKERRQ(ierr);
        }
        if (trans && N[k] > 1) {
          ierr = PetscSNPrintf(stage_s, sizeof(stage_s), "trans_type_%s-bs_%D-N_%02d", type[i], bs[j], (int)N[k]);CHKERRQ(ierr);
          ierr = PetscLogStageRegister(stage_s, &tstage);CHKERRQ(ierr);
        }

        /* A*B */
        if (N[k] > 1) {
          if (!maij) {
            ierr = MatProductCreateWithMat(A, C, NULL, D);CHKERRQ(ierr);
            ierr = MatProductSetType(D, MATPRODUCT_AB);CHKERRQ(ierr);
            ierr = MatProductSetFromOptions(D);CHKERRQ(ierr);
            ierr = MatProductSymbolic(D);CHKERRQ(ierr);
          }

          if (!mkl) {
            if (!maij) {
              ierr = MatProductNumeric(D);CHKERRQ(ierr);
              ierr = PetscPrintf(PETSC_COMM_WORLD, "Benchmarking MatProduct %s: with A %s %Dx%D and B %s %Dx%D\n", MatProductTypes[MATPRODUCT_AB], Atype, AM, AN, Ctype, CM, CN);
              ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
              for (t = 0; t < trial; ++t) {
                ierr = MatProductNumeric(D);CHKERRQ(ierr);
              }
              ierr = PetscLogStagePop();CHKERRQ(ierr);
            } else {
              Mat               E, Ct, Dt;
              Vec               cC, cD;
              const PetscScalar *c_ptr;
              PetscScalar       *d_ptr;
              ierr = MatCreateMAIJ(A, N[k], &E);CHKERRQ(ierr);
              ierr = MatDenseGetLocalMatrix(C, &Ct);CHKERRQ(ierr);
              ierr = MatDenseGetLocalMatrix(D, &Dt);CHKERRQ(ierr);
              ierr = MatTranspose(Ct, MAT_INPLACE_MATRIX, &Ct);CHKERRQ(ierr);
              ierr = MatTranspose(Dt, MAT_INPLACE_MATRIX, &Dt);CHKERRQ(ierr);
              ierr = MatDenseGetArrayRead(Ct, &c_ptr);CHKERRQ(ierr);
              ierr = MatDenseGetArrayWrite(Dt, &d_ptr);CHKERRQ(ierr);
              ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, AM * N[k], PETSC_DECIDE, c_ptr, &cC);CHKERRQ(ierr);
              ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, AM * N[k], PETSC_DECIDE, d_ptr, &cD);CHKERRQ(ierr);
              ierr = MatMult(E, cC, cD);CHKERRQ(ierr);
              ierr = PetscPrintf(PETSC_COMM_WORLD, "Benchmarking MatMult: with A %s %Dx%D and B %s %Dx%D\n", MATMAIJ, AM, AN, VECMPI, AM * N[k], 1);
              ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
              for (t = 0; t < trial; ++t) {
                ierr = MatMult(E, cC, cD);CHKERRQ(ierr);
              }
              ierr = PetscLogStagePop();CHKERRQ(ierr);
              ierr = VecDestroy(&cD);CHKERRQ(ierr);
              ierr = VecDestroy(&cC);CHKERRQ(ierr);
              ierr = MatDestroy(&E);CHKERRQ(ierr);
              ierr = MatDenseRestoreArrayWrite(Dt, &d_ptr);CHKERRQ(ierr);
              ierr = MatDenseRestoreArrayRead(Ct, &c_ptr);CHKERRQ(ierr);
              ierr = MatTranspose(Ct, MAT_INPLACE_MATRIX, &Ct);CHKERRQ(ierr);
              ierr = MatTranspose(Dt, MAT_INPLACE_MATRIX, &Dt);CHKERRQ(ierr);
            }
          } else {
            const PetscScalar *c_ptr;
            PetscScalar       *d_ptr;

            ierr = MatDenseGetArrayRead(C, &c_ptr);CHKERRQ(ierr);
            ierr = MatDenseGetArrayWrite(D, &d_ptr);CHKERRQ(ierr);
            PetscStackCallMKLSparse(mkl_sparse_d_mm,(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, spr, descr, SPARSE_LAYOUT_COLUMN_MAJOR, c_ptr, CN, CM, 0.0, d_ptr, CM));
            ierr = PetscPrintf(PETSC_COMM_WORLD, "Benchmarking mkl_sparse_d_mm (COLUMN_MAJOR): with A %s %Dx%D and B %s %Dx%D\n", Atype, AM, AN, Ctype, CM, CN);
            ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
            for (t = 0; t < trial; ++t) {
              PetscStackCallMKLSparse(mkl_sparse_d_mm, (SPARSE_OPERATION_NON_TRANSPOSE, 1.0, spr, descr, SPARSE_LAYOUT_COLUMN_MAJOR, c_ptr, CN, CM, 0.0, d_ptr, CM));
            }
            ierr = PetscLogStagePop();CHKERRQ(ierr);
            ierr = MatDenseRestoreArrayWrite(D, &d_ptr);CHKERRQ(ierr);
            ierr = MatDenseRestoreArrayRead(C, &c_ptr);CHKERRQ(ierr);
          }
        } else if (maij) {
          ierr = MatDestroy(&C);CHKERRQ(ierr);
          ierr = MatDestroy(&D);CHKERRQ(ierr);
          continue;
        } else if (!mkl) {
          Vec cC, cD;

          ierr = MatDenseGetColumnVecRead(C, 0, &cC);CHKERRQ(ierr);
          ierr = MatDenseGetColumnVecWrite(D, 0, &cD);CHKERRQ(ierr);
          ierr = MatMult(A, cC, cD);CHKERRQ(ierr);
          ierr = PetscPrintf(PETSC_COMM_WORLD, "Benchmarking MatMult: with A %s %Dx%D\n", Atype, AM, AN);
          ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
          for (t = 0; t < trial; ++t) {
            ierr = MatMult(A, cC, cD);CHKERRQ(ierr);
          }
          ierr = PetscLogStagePop();CHKERRQ(ierr);
          ierr = MatDenseRestoreColumnVecRead(C, 0, &cC);CHKERRQ(ierr);
          ierr = MatDenseRestoreColumnVecWrite(D, 0, &cD);CHKERRQ(ierr);
        } else {
          const PetscScalar *c_ptr;
          PetscScalar       *d_ptr;

          ierr = MatDenseGetArrayRead(C, &c_ptr);CHKERRQ(ierr);
          ierr = MatDenseGetArrayWrite(D, &d_ptr);CHKERRQ(ierr);
          ierr = PetscPrintf(PETSC_COMM_WORLD, "Benchmarking mkl_sparse_d_mv: with A %s %Dx%D\n", Atype, AM, AN);
          PetscStackCallMKLSparse(mkl_sparse_d_mv, (SPARSE_OPERATION_NON_TRANSPOSE, 1.0, spr, descr, c_ptr, 0.0, d_ptr));
          ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
          for (t = 0; t < trial; ++t) {
            PetscStackCallMKLSparse(mkl_sparse_d_mv, (SPARSE_OPERATION_NON_TRANSPOSE, 1.0, spr, descr, c_ptr, 0.0, d_ptr));
          }
          ierr = PetscLogStagePop();CHKERRQ(ierr);
          ierr = MatDenseRestoreArrayWrite(D, &d_ptr);CHKERRQ(ierr);
          ierr = MatDenseRestoreArrayRead(C, &c_ptr);CHKERRQ(ierr);
        }

        if (check) {
          ierr = MatMatMultEqual(A, C, D, 10, &flg);CHKERRQ(ierr);
          if (!flg) {
            MatType Dtype;

            ierr = MatGetType(D, &Dtype);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD, "Error with A %s%s, C %s, D %s, Nk %D\n", Atype, mkl ? "mkl" : "", Ctype, Dtype, N[k]);CHKERRQ(ierr);
          }
        }

        /* MKL implementation seems buggy for ABt */
        /* A*Bt */
        if (!mkl && trans && N[k] > 1) {
          ierr = PetscObjectTypeCompareAny((PetscObject)A, &flg, MATSEQAIJ, MATMPIAIJ, "");CHKERRQ(ierr);
          if (flg) {
            ierr = MatTranspose(C, MAT_INPLACE_MATRIX, &C);CHKERRQ(ierr);
            ierr = MatGetType(C, &Ctype);CHKERRQ(ierr);
            if (!mkl) {
              ierr = MatProductCreateWithMat(A, C, NULL, D);CHKERRQ(ierr);
              ierr = MatProductSetType(D, MATPRODUCT_ABt);CHKERRQ(ierr);
              ierr = MatProductSetFromOptions(D);CHKERRQ(ierr);
              ierr = MatProductSymbolic(D);CHKERRQ(ierr);
              ierr = MatProductNumeric(D);CHKERRQ(ierr);
              ierr = PetscPrintf(PETSC_COMM_WORLD, "Benchmarking MatProduct %s: with A %s %Dx%D and Bt %s %Dx%D\n", MatProductTypes[MATPRODUCT_ABt], Atype, AM, AN, Ctype, CM, CN);
              ierr = PetscLogStagePush(tstage);CHKERRQ(ierr);
              for (t = 0; t < trial; ++t) {
                ierr = MatProductNumeric(D);CHKERRQ(ierr);
              }
              ierr = PetscLogStagePop();CHKERRQ(ierr);
            } else {
              const PetscScalar *c_ptr;
              PetscScalar       *d_ptr;

              PetscStackCallMKLSparse(mkl_sparse_set_mm_hint, (spr, SPARSE_OPERATION_NON_TRANSPOSE, descr, SPARSE_LAYOUT_ROW_MAJOR, N[k], 1 + trial));
              PetscStackCallMKLSparse(mkl_sparse_optimize, (spr));
              ierr = MatDenseGetArrayRead(C, &c_ptr);CHKERRQ(ierr);
              ierr = MatDenseGetArrayWrite(D, &d_ptr);CHKERRQ(ierr);
              ierr = PetscPrintf(PETSC_COMM_WORLD, "Benchmarking mkl_sparse_d_mm (ROW_MAJOR): with A %s %Dx%D and B %s %Dx%D\n", Atype, AM, AN, Ctype, CM, CN);
              PetscStackCallMKLSparse(mkl_sparse_d_mm, (SPARSE_OPERATION_NON_TRANSPOSE, 1.0, spr, descr, SPARSE_LAYOUT_ROW_MAJOR, c_ptr, CN, CM, 0.0, d_ptr, CM));
              ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
              for (t = 0; t < trial; ++t) {
                PetscStackCallMKLSparse(mkl_sparse_d_mm, (SPARSE_OPERATION_NON_TRANSPOSE, 1.0, spr, descr, SPARSE_LAYOUT_ROW_MAJOR, c_ptr, CN, CM, 0.0, d_ptr, CM));
              }
              ierr = PetscLogStagePop();CHKERRQ(ierr);
              ierr = MatDenseRestoreArrayWrite(D, &d_ptr);CHKERRQ(ierr);
              ierr = MatDenseRestoreArrayRead(C, &c_ptr);CHKERRQ(ierr);
            }
          }
        }

        if (!mkl && trans && N[k] > 1 && flg && check) {
          ierr = MatMatTransposeMultEqual(A, C, D, 10, &flg);CHKERRQ(ierr);
          if (!flg) {
            MatType Dtype;
            ierr = MatGetType(D, &Dtype);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD, "Error with A %s%s, C %s, D %s, Nk %D\n", Atype, mkl ? "mkl" : "", Ctype, Dtype, N[k]);CHKERRQ(ierr);
          }
        }
        ierr = MatDestroy(&C);CHKERRQ(ierr);
        ierr = MatDestroy(&D);CHKERRQ(ierr);
      }
      if (mkl) {
        PetscStackCallMKLSparse(mkl_sparse_destroy, (spr));
        ierr = PetscFree(ia_ptr);CHKERRQ(ierr);
        ierr = PetscFree(ja_ptr);CHKERRQ(ierr);
      }
      if (cuda && i != ntype - 1) {
        ierr = PetscPrintf(PETSC_COMM_WORLD, "AIJCUSPARSE must be last, otherwise MatConvert() to another MatType is too slow\n");CHKERRQ(ierr);
        break;
      }
    }
    if (E != A) {
      ierr = MatDestroy(&E);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }
  for (m = 0; m < ntype; ++m) {
    ierr = PetscFree(type[m]);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return 0;
}

/*TEST

   testset:
     nsize: 1
     requires: double !complex !define(PETSC_USE_64BIT_INDICES)
     filter: sed "/Benchmarking/d"
     args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/spd-real-int32-float64 -bs 1,2,3 -N 1,2,18 -check -trans -convert_aij {{false true}shared output}
     test:
       suffix: basic
       args: -type aij,sbaij,baij
       output_file: output/ex237.out
     test:
       suffix: maij
       args: -type aij,maij
       output_file: output/ex237.out
     test:
       suffix: cuda
       requires: cuda
       args: -type aij,aijcusparse
       output_file: output/ex237.out
     test:
       suffix: mkl
       requires: mkl
       args: -type aij,aijmkl,baijmkl,sbaijmkl
       output_file: output/ex237.out

TEST*/
