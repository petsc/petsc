static char help[] = "Mini-app to benchmark matrix--matrix multiplication\n\n";

/*
  See the paper below for more information

   "KSPHPDDM and PCHPDDM: Extending PETSc with Robust Overlapping Schwarz Preconditioners and Advanced Krylov Methods",
   P. Jolivet, J. E. Roman, and S. Zampini (2020).
*/

#include <petsc.h>

#if defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
#include <mkl.h>
#define PetscStackCallMKLSparse(func, args) do {               \
    sparse_status_t __ierr;                                    \
    PetscStackPush(#func);                                     \
    __ierr = func args;                                        \
    PetscStackPop;                                             \
    PetscCheckFalse(__ierr != SPARSE_STATUS_SUCCESS,PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in %s(): error code %d", #func, (int)__ierr); \
  } while (0)
#else
#define PetscStackCallMKLSparse(func, args) do {               \
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No MKL support"); \
  } while (0)
#endif

int main(int argc, char** argv)
{
  Mat          A, C, D, E;
  PetscInt     nbs             = 10, ntype = 10, nN = 8, m, M, trial = 5;
  PetscViewer  viewer;
  PetscInt     bs[10], N[8];
  char        *type[10];
  PetscMPIInt  size;
  PetscBool    flg, cuda, maij = PETSC_FALSE, check = PETSC_FALSE, trans = PETSC_FALSE, convert = PETSC_FALSE, mkl;
  char         file[PETSC_MAX_PATH_LEN];

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1,PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only");
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", file, PETSC_MAX_PATH_LEN, &flg));
  PetscCheck(flg,PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Must indicate binary file with the -f option");
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-trial", &trial, NULL));
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-bs", bs, &nbs, &flg));
  if (!flg) {
    nbs = 1;
    bs[0] = 1;
  }
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-N", N, &nN, &flg));
  if (!flg) {
    nN = 8;
    N[0] = 1;  N[1] = 2;  N[2] = 4;  N[3] = 8;
    N[4] = 16; N[5] = 32; N[6] = 64; N[7] = 128;
  }
  PetscCall(PetscOptionsGetStringArray(NULL, NULL, "-type", type, &ntype, &flg));
  if (!flg) {
    ntype = 1;
    PetscCall(PetscStrallocpy(MATSEQAIJ, &type[0]));
  }
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-check", &check, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-trans", &trans, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-convert_aij", &convert, NULL));
  for (PetscInt j = 0; j < nbs; ++j) {
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &viewer));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatLoad(A, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &flg, MATSEQAIJ, MATMPIAIJ, ""));
    PetscCheck(flg,PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Must indicate a MatAIJ input matrix");
    PetscCall(MatGetSize(A, &m, &M));
    if (m == M) {
      Mat oA;
      PetscCall(MatTranspose(A, MAT_INITIAL_MATRIX, &oA));
      PetscCall(MatAXPY(A, 1.0, oA, DIFFERENT_NONZERO_PATTERN));
      PetscCall(MatDestroy(&oA));
    }
    PetscCall(MatGetLocalSize(A, &m, NULL));
    PetscCall(MatGetSize(A, &M, NULL));
    if (bs[j] > 1) {
      Mat               T, Tt, B;
      const PetscScalar *ptr;
      PetscScalar       *val, *Aa;
      const PetscInt    *Ai, *Aj;
      PetscInt          An, i, k;
      PetscBool         done;

      PetscCall(MatCreateDense(PETSC_COMM_SELF, bs[j], bs[j], bs[j], bs[j], NULL, &T));
      PetscCall(MatSetRandom(T, NULL));
      PetscCall(MatTranspose(T, MAT_INITIAL_MATRIX, &Tt));
      PetscCall(MatAXPY(T, 1.0, Tt, SAME_NONZERO_PATTERN));
      PetscCall(MatDestroy(&Tt));
      PetscCall(MatDenseGetArrayRead(T, &ptr));
      PetscCall(MatGetRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &An, &Ai, &Aj, &done));
      PetscCheckFalse(!done || An != m,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Inconsistent sizes");
      PetscCall(MatSeqAIJGetArray(A, &Aa));
      PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
      PetscCall(MatSetType(B, MATSEQBAIJ));
      PetscCall(MatSetSizes(B, bs[j] * An, bs[j] * An, PETSC_DECIDE, PETSC_DECIDE));
      PetscCall(PetscMalloc1(Ai[An] * bs[j] * bs[j], &val));
      for (i = 0; i < Ai[An]; ++i)
        for (k = 0; k < bs[j] * bs[j]; ++k)
          val[i * bs[j] * bs[j] + k] = Aa[i] * ptr[k];
      PetscCall(MatSetOption(B, MAT_ROW_ORIENTED, PETSC_FALSE));
      PetscCall(MatSeqBAIJSetPreallocationCSR(B, bs[j], Ai, Aj, val));
      PetscCall(PetscFree(val));
      PetscCall(MatSeqAIJRestoreArray(A, &Aa));
      PetscCall(MatRestoreRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &An, &Ai, &Aj, &done));
      PetscCall(MatDenseRestoreArrayRead(T, &ptr));
      PetscCall(MatDestroy(&T));
      PetscCall(MatDestroy(&A));
      A    = B;
    }
    /* reconvert back to SeqAIJ before converting to the desired type later */
    if (!convert) E = A;
    PetscCall(MatConvert(A, MATSEQAIJ, convert ? MAT_INITIAL_MATRIX : MAT_INPLACE_MATRIX, &E));
    PetscCall(MatSetOption(E, MAT_SYMMETRIC, PETSC_TRUE));
    for (PetscInt i = 0; i < ntype; ++i) {
      char        *tmp;
      PetscInt    *ia_ptr, *ja_ptr, k;
      PetscScalar *a_ptr;
#if defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
      struct matrix_descr descr;
      sparse_matrix_t     spr;
      descr.type = SPARSE_MATRIX_TYPE_GENERAL;
      descr.diag = SPARSE_DIAG_NON_UNIT;
#endif
      if (convert) {
        PetscCall(MatDestroy(&A));
      }
      PetscCall(PetscStrstr(type[i], "mkl", &tmp));
      if (tmp) {
        size_t mlen, tlen;
        char base[256];

        mkl  = PETSC_TRUE;
        PetscCall(PetscStrlen(tmp, &mlen));
        PetscCall(PetscStrlen(type[i], &tlen));
        PetscCall(PetscStrncpy(base, type[i], tlen-mlen + 1));
        PetscCall(MatConvert(E, base, convert ? MAT_INITIAL_MATRIX : MAT_INPLACE_MATRIX, &A));
      } else {
        mkl  = PETSC_FALSE;
        PetscCall(PetscStrstr(type[i], "maij", &tmp));
        if (!tmp) {
          PetscCall(MatConvert(E, type[i], convert ? MAT_INITIAL_MATRIX : MAT_INPLACE_MATRIX, &A));
        } else {
          PetscCall(MatConvert(E, MATAIJ, convert ? MAT_INITIAL_MATRIX : MAT_INPLACE_MATRIX, &A));
          maij = PETSC_TRUE;
        }
      }
      PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &cuda, MATSEQAIJCUSPARSE, MATMPIAIJCUSPARSE, ""));
      if (mkl) {
        const PetscInt *Ai, *Aj;
        PetscInt       An;
        PetscBool      done;

        PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &flg, MATSEQAIJ, MATSEQBAIJ, MATSEQSBAIJ, ""));
        PetscCheck(flg,PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Not implemented");
        PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJ, &flg));
        PetscCall(MatGetRowIJ(A, 0, PETSC_FALSE, flg ? PETSC_FALSE : PETSC_TRUE, &An, &Ai, &Aj, &done));
        PetscCheck(done,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Inconsistent sizes");
        PetscCall(PetscMalloc1(An + 1, &ia_ptr));
        PetscCall(PetscMalloc1(Ai[An], &ja_ptr));
        if (flg) { /* SeqAIJ */
          for (k = 0; k < An + 1; ++k) ia_ptr[k] = Ai[k];
          for (k = 0; k < Ai[An]; ++k) ja_ptr[k] = Aj[k];
          PetscCall(MatSeqAIJGetArray(A, &a_ptr));
          PetscStackCallMKLSparse(mkl_sparse_d_create_csr, (&spr, SPARSE_INDEX_BASE_ZERO, An, An, ia_ptr, ia_ptr + 1, ja_ptr, a_ptr));
        } else {
          PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQBAIJ, &flg));
          if (flg) {
            for (k = 0; k < An + 1; ++k) ia_ptr[k] = Ai[k] + 1; /* Fortran indexing to maximize cases covered by _mm routines */
            for (k = 0; k < Ai[An]; ++k) ja_ptr[k] = Aj[k] + 1; /* Fortran indexing to maximize cases covered by _mm routines */
            PetscCall(MatSeqBAIJGetArray(A, &a_ptr));
            PetscStackCallMKLSparse(mkl_sparse_d_create_bsr, (&spr, SPARSE_INDEX_BASE_ONE, SPARSE_LAYOUT_COLUMN_MAJOR, An, An, bs[j], ia_ptr, ia_ptr + 1, ja_ptr, a_ptr));
          } else {
            PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQSBAIJ, &flg));
            if (flg) {
              for (k = 0; k < An + 1; ++k) ia_ptr[k] = Ai[k] + 1; /* Fortran indexing to maximize cases covered by _mm routines */
              for (k = 0; k < Ai[An]; ++k) ja_ptr[k] = Aj[k] + 1; /* Fortran indexing to maximize cases covered by _mm routines */
              PetscCall(MatSeqSBAIJGetArray(A, &a_ptr));
              PetscStackCallMKLSparse(mkl_sparse_d_create_bsr, (&spr, SPARSE_INDEX_BASE_ONE, SPARSE_LAYOUT_COLUMN_MAJOR, An, An, bs[j], ia_ptr, ia_ptr + 1, ja_ptr, a_ptr));
#if defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
              descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
              descr.mode = SPARSE_FILL_MODE_UPPER;
              descr.diag = SPARSE_DIAG_NON_UNIT;
#endif
            }
          }
        }
        PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJ, &flg));
        PetscCall(MatRestoreRowIJ(A, 0, PETSC_FALSE, flg ? PETSC_FALSE : PETSC_TRUE, &An, &Ai, &Aj, &done));
      }

      PetscCall(MatViewFromOptions(A, NULL, "-A_view"));

      for (k = 0; k < nN; ++k) {
        MatType       Atype, Ctype;
        PetscInt      AM, AN, CM, CN, t;
#if defined(PETSC_USE_LOG)
        PetscLogStage stage, tstage;
        char          stage_s[256];
#endif

        PetscCall(MatCreateDense(PETSC_COMM_WORLD, bs[j] * m, PETSC_DECIDE, bs[j] * M, N[k], NULL, &C));
        PetscCall(MatCreateDense(PETSC_COMM_WORLD, bs[j] * m, PETSC_DECIDE, bs[j] * M, N[k], NULL, &D));
        PetscCall(MatSetRandom(C, NULL));
        if (cuda) { /* convert to GPU if needed */
          PetscCall(MatConvert(C, MATDENSECUDA, MAT_INPLACE_MATRIX, &C));
          PetscCall(MatConvert(D, MATDENSECUDA, MAT_INPLACE_MATRIX, &D));
        }
        if (mkl) {
          if (N[k] > 1) PetscStackCallMKLSparse(mkl_sparse_set_mm_hint, (spr, SPARSE_OPERATION_NON_TRANSPOSE, descr, SPARSE_LAYOUT_COLUMN_MAJOR, N[k], 1 + trial));
          else          PetscStackCallMKLSparse(mkl_sparse_set_mv_hint, (spr, SPARSE_OPERATION_NON_TRANSPOSE, descr, 1 + trial));
          PetscStackCallMKLSparse(mkl_sparse_set_memory_hint, (spr, SPARSE_MEMORY_AGGRESSIVE));
          PetscStackCallMKLSparse(mkl_sparse_optimize, (spr));
        }
        PetscCall(MatGetType(A, &Atype));
        PetscCall(MatGetType(C, &Ctype));
        PetscCall(MatGetSize(A, &AM, &AN));
        PetscCall(MatGetSize(C, &CM, &CN));

#if defined(PETSC_USE_LOG)
        if (!maij || N[k] > 1) {
          PetscCall(PetscSNPrintf(stage_s, sizeof(stage_s), "type_%s-bs_%" PetscInt_FMT "-N_%02d", type[i], bs[j], (int)N[k]));
          PetscCall(PetscLogStageRegister(stage_s, &stage));
        }
        if (trans && N[k] > 1) {
          PetscCall(PetscSNPrintf(stage_s, sizeof(stage_s), "trans_type_%s-bs_%" PetscInt_FMT "-N_%02d", type[i], bs[j], (int)N[k]));
          PetscCall(PetscLogStageRegister(stage_s, &tstage));
        }
#endif
        /* A*B */
        if (N[k] > 1) {
          if (!maij) {
            PetscCall(MatProductCreateWithMat(A, C, NULL, D));
            PetscCall(MatProductSetType(D, MATPRODUCT_AB));
            PetscCall(MatProductSetFromOptions(D));
            PetscCall(MatProductSymbolic(D));
          }

          if (!mkl) {
            if (!maij) {
              PetscCall(MatProductNumeric(D));
              PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Benchmarking MatProduct %s: with A %s %" PetscInt_FMT "x%" PetscInt_FMT " and B %s %" PetscInt_FMT "x%" PetscInt_FMT "\n", MatProductTypes[MATPRODUCT_AB], Atype, AM, AN, Ctype, CM, CN));
              PetscCall(PetscLogStagePush(stage));
              for (t = 0; t < trial; ++t) {
                PetscCall(MatProductNumeric(D));
              }
              PetscCall(PetscLogStagePop());
            } else {
              Mat               E, Ct, Dt;
              Vec               cC, cD;
              const PetscScalar *c_ptr;
              PetscScalar       *d_ptr;
              PetscCall(MatCreateMAIJ(A, N[k], &E));
              PetscCall(MatDenseGetLocalMatrix(C, &Ct));
              PetscCall(MatDenseGetLocalMatrix(D, &Dt));
              PetscCall(MatTranspose(Ct, MAT_INPLACE_MATRIX, &Ct));
              PetscCall(MatTranspose(Dt, MAT_INPLACE_MATRIX, &Dt));
              PetscCall(MatDenseGetArrayRead(Ct, &c_ptr));
              PetscCall(MatDenseGetArrayWrite(Dt, &d_ptr));
              PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, AM * N[k], PETSC_DECIDE, c_ptr, &cC));
              PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, AM * N[k], PETSC_DECIDE, d_ptr, &cD));
              PetscCall(MatMult(E, cC, cD));
              PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Benchmarking MatMult: with A %s %" PetscInt_FMT "x%" PetscInt_FMT " and B %s %" PetscInt_FMT "x%" PetscInt_FMT "\n", MATMAIJ, AM, AN, VECMPI, AM * N[k], 1));
              PetscCall(PetscLogStagePush(stage));
              for (t = 0; t < trial; ++t) {
                PetscCall(MatMult(E, cC, cD));
              }
              PetscCall(PetscLogStagePop());
              PetscCall(VecDestroy(&cD));
              PetscCall(VecDestroy(&cC));
              PetscCall(MatDestroy(&E));
              PetscCall(MatDenseRestoreArrayWrite(Dt, &d_ptr));
              PetscCall(MatDenseRestoreArrayRead(Ct, &c_ptr));
              PetscCall(MatTranspose(Ct, MAT_INPLACE_MATRIX, &Ct));
              PetscCall(MatTranspose(Dt, MAT_INPLACE_MATRIX, &Dt));
            }
          } else {
            const PetscScalar *c_ptr;
            PetscScalar       *d_ptr;

            PetscCall(MatDenseGetArrayRead(C, &c_ptr));
            PetscCall(MatDenseGetArrayWrite(D, &d_ptr));
            PetscStackCallMKLSparse(mkl_sparse_d_mm,(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, spr, descr, SPARSE_LAYOUT_COLUMN_MAJOR, c_ptr, CN, CM, 0.0, d_ptr, CM));
            PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Benchmarking mkl_sparse_d_mm (COLUMN_MAJOR): with A %s %" PetscInt_FMT "x%" PetscInt_FMT " and B %s %" PetscInt_FMT "x%" PetscInt_FMT "\n", Atype, AM, AN, Ctype, CM, CN));
            PetscCall(PetscLogStagePush(stage));
            for (t = 0; t < trial; ++t) {
              PetscStackCallMKLSparse(mkl_sparse_d_mm, (SPARSE_OPERATION_NON_TRANSPOSE, 1.0, spr, descr, SPARSE_LAYOUT_COLUMN_MAJOR, c_ptr, CN, CM, 0.0, d_ptr, CM));
            }
            PetscCall(PetscLogStagePop());
            PetscCall(MatDenseRestoreArrayWrite(D, &d_ptr));
            PetscCall(MatDenseRestoreArrayRead(C, &c_ptr));
          }
        } else if (maij) {
          PetscCall(MatDestroy(&C));
          PetscCall(MatDestroy(&D));
          continue;
        } else if (!mkl) {
          Vec cC, cD;

          PetscCall(MatDenseGetColumnVecRead(C, 0, &cC));
          PetscCall(MatDenseGetColumnVecWrite(D, 0, &cD));
          PetscCall(MatMult(A, cC, cD));
          PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Benchmarking MatMult: with A %s %" PetscInt_FMT "x%" PetscInt_FMT "\n", Atype, AM, AN));
          PetscCall(PetscLogStagePush(stage));
          for (t = 0; t < trial; ++t) {
            PetscCall(MatMult(A, cC, cD));
          }
          PetscCall(PetscLogStagePop());
          PetscCall(MatDenseRestoreColumnVecRead(C, 0, &cC));
          PetscCall(MatDenseRestoreColumnVecWrite(D, 0, &cD));
        } else {
          const PetscScalar *c_ptr;
          PetscScalar       *d_ptr;

          PetscCall(MatDenseGetArrayRead(C, &c_ptr));
          PetscCall(MatDenseGetArrayWrite(D, &d_ptr));
          PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Benchmarking mkl_sparse_d_mv: with A %s %" PetscInt_FMT "x%" PetscInt_FMT "\n", Atype, AM, AN));
          PetscStackCallMKLSparse(mkl_sparse_d_mv, (SPARSE_OPERATION_NON_TRANSPOSE, 1.0, spr, descr, c_ptr, 0.0, d_ptr));
          PetscCall(PetscLogStagePush(stage));
          for (t = 0; t < trial; ++t) {
            PetscStackCallMKLSparse(mkl_sparse_d_mv, (SPARSE_OPERATION_NON_TRANSPOSE, 1.0, spr, descr, c_ptr, 0.0, d_ptr));
          }
          PetscCall(PetscLogStagePop());
          PetscCall(MatDenseRestoreArrayWrite(D, &d_ptr));
          PetscCall(MatDenseRestoreArrayRead(C, &c_ptr));
        }

        if (check) {
          PetscCall(MatMatMultEqual(A, C, D, 10, &flg));
          if (!flg) {
            MatType Dtype;

            PetscCall(MatGetType(D, &Dtype));
            PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error with A %s%s, C %s, D %s, Nk %" PetscInt_FMT "\n", Atype, mkl ? "mkl" : "", Ctype, Dtype, N[k]));
          }
        }

        /* MKL implementation seems buggy for ABt */
        /* A*Bt */
        if (!mkl && trans && N[k] > 1) {
          PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &flg, MATSEQAIJ, MATMPIAIJ, ""));
          if (flg) {
            PetscCall(MatTranspose(C, MAT_INPLACE_MATRIX, &C));
            PetscCall(MatGetType(C, &Ctype));
            if (!mkl) {
              PetscCall(MatProductCreateWithMat(A, C, NULL, D));
              PetscCall(MatProductSetType(D, MATPRODUCT_ABt));
              PetscCall(MatProductSetFromOptions(D));
              PetscCall(MatProductSymbolic(D));
              PetscCall(MatProductNumeric(D));
              PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Benchmarking MatProduct %s: with A %s %" PetscInt_FMT "x%" PetscInt_FMT " and Bt %s %" PetscInt_FMT "x%" PetscInt_FMT "\n", MatProductTypes[MATPRODUCT_ABt], Atype, AM, AN, Ctype, CM, CN));
              PetscCall(PetscLogStagePush(tstage));
              for (t = 0; t < trial; ++t) {
                PetscCall(MatProductNumeric(D));
              }
              PetscCall(PetscLogStagePop());
            } else {
              const PetscScalar *c_ptr;
              PetscScalar       *d_ptr;

              PetscStackCallMKLSparse(mkl_sparse_set_mm_hint, (spr, SPARSE_OPERATION_NON_TRANSPOSE, descr, SPARSE_LAYOUT_ROW_MAJOR, N[k], 1 + trial));
              PetscStackCallMKLSparse(mkl_sparse_optimize, (spr));
              PetscCall(MatDenseGetArrayRead(C, &c_ptr));
              PetscCall(MatDenseGetArrayWrite(D, &d_ptr));
              PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Benchmarking mkl_sparse_d_mm (ROW_MAJOR): with A %s %" PetscInt_FMT "x%" PetscInt_FMT " and B %s %" PetscInt_FMT "x%" PetscInt_FMT "\n", Atype, AM, AN, Ctype, CM, CN));
              PetscStackCallMKLSparse(mkl_sparse_d_mm, (SPARSE_OPERATION_NON_TRANSPOSE, 1.0, spr, descr, SPARSE_LAYOUT_ROW_MAJOR, c_ptr, CN, CM, 0.0, d_ptr, CM));
              PetscCall(PetscLogStagePush(stage));
              for (t = 0; t < trial; ++t) {
                PetscStackCallMKLSparse(mkl_sparse_d_mm, (SPARSE_OPERATION_NON_TRANSPOSE, 1.0, spr, descr, SPARSE_LAYOUT_ROW_MAJOR, c_ptr, CN, CM, 0.0, d_ptr, CM));
              }
              PetscCall(PetscLogStagePop());
              PetscCall(MatDenseRestoreArrayWrite(D, &d_ptr));
              PetscCall(MatDenseRestoreArrayRead(C, &c_ptr));
            }
          }
        }

        if (!mkl && trans && N[k] > 1 && flg && check) {
          PetscCall(MatMatTransposeMultEqual(A, C, D, 10, &flg));
          if (!flg) {
            MatType Dtype;
            PetscCall(MatGetType(D, &Dtype));
            PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error with A %s%s, C %s, D %s, Nk %" PetscInt_FMT "\n", Atype, mkl ? "mkl" : "", Ctype, Dtype, N[k]));
          }
        }
        PetscCall(MatDestroy(&C));
        PetscCall(MatDestroy(&D));
      }
      if (mkl) {
        PetscStackCallMKLSparse(mkl_sparse_destroy, (spr));
        PetscCall(PetscFree(ia_ptr));
        PetscCall(PetscFree(ja_ptr));
      }
      if (cuda && i != ntype - 1) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "AIJCUSPARSE must be last, otherwise MatConvert() to another MatType is too slow\n"));
        break;
      }
    }
    if (E != A) PetscCall(MatDestroy(&E));
    PetscCall(MatDestroy(&A));
  }
  for (m = 0; m < ntype; ++m) PetscCall(PetscFree(type[m]));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
   build:
     requires: double !complex !defined(PETSC_USE_64BIT_INDICES)

   testset:
     nsize: 1
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
       requires: mkl_sparse_optimize
       args: -type aij,aijmkl,baijmkl,sbaijmkl
       output_file: output/ex237.out

TEST*/
