
static char help[] = "Tests converting a matrix to another format with MatConvert().\n\n";

#include <petscmat.h>
/* Usage: mpiexec -n <np> ex55 -verbose <0 or 1> */

int main(int argc, char **args)
{
  Mat         C, A, B, D;
  PetscInt    i, j, ntypes, bs, mbs, m, block, d_nz = 6, o_nz = 3, col[3], row, verbose = 0;
  PetscMPIInt size, rank;
  MatType     type[9];
  char        file[PETSC_MAX_PATH_LEN];
  PetscViewer fd;
  PetscBool   equal, flg_loadmat, flg, issymmetric;
  PetscScalar value[3];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-verbose", &verbose, NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", file, sizeof(file), &flg_loadmat));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCall(PetscOptionsHasName(NULL, NULL, "-testseqaij", &flg));
  if (flg) {
    if (size == 1) {
      type[0] = MATSEQAIJ;
    } else {
      type[0] = MATMPIAIJ;
    }
  } else {
    type[0] = MATAIJ;
  }
  if (size == 1) {
    ntypes  = 3;
    type[1] = MATSEQBAIJ;
    type[2] = MATSEQSBAIJ;
  } else {
    ntypes  = 3;
    type[1] = MATMPIBAIJ;
    type[2] = MATMPISBAIJ;
  }

  /* input matrix C */
  if (flg_loadmat) {
    /* Open binary file. */
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));

    /* Load a baij matrix, then destroy the viewer. */
    PetscCall(MatCreate(PETSC_COMM_WORLD, &C));
    if (size == 1) {
      PetscCall(MatSetType(C, MATSEQBAIJ));
    } else {
      PetscCall(MatSetType(C, MATMPIBAIJ));
    }
    PetscCall(MatSetFromOptions(C));
    PetscCall(MatLoad(C, fd));
    PetscCall(PetscViewerDestroy(&fd));
  } else { /* Create a baij mat with bs>1  */
    bs  = 2;
    mbs = 8;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-mbs", &mbs, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-bs", &bs, NULL));
    PetscCheck(bs > 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, " bs must be >1 in this case");
    m = mbs * bs;
    PetscCall(MatCreateBAIJ(PETSC_COMM_WORLD, bs, PETSC_DECIDE, PETSC_DECIDE, m, m, d_nz, NULL, o_nz, NULL, &C));
    for (block = 0; block < mbs; block++) {
      /* diagonal blocks */
      value[0] = -1.0;
      value[1] = 4.0;
      value[2] = -1.0;
      for (i = 1 + block * bs; i < bs - 1 + block * bs; i++) {
        col[0] = i - 1;
        col[1] = i;
        col[2] = i + 1;
        PetscCall(MatSetValues(C, 1, &i, 3, col, value, INSERT_VALUES));
      }
      i        = bs - 1 + block * bs;
      col[0]   = bs - 2 + block * bs;
      col[1]   = bs - 1 + block * bs;
      value[0] = -1.0;
      value[1] = 4.0;
      PetscCall(MatSetValues(C, 1, &i, 2, col, value, INSERT_VALUES));

      i        = 0 + block * bs;
      col[0]   = 0 + block * bs;
      col[1]   = 1 + block * bs;
      value[0] = 4.0;
      value[1] = -1.0;
      PetscCall(MatSetValues(C, 1, &i, 2, col, value, INSERT_VALUES));
    }
    /* off-diagonal blocks */
    value[0] = -1.0;
    for (i = 0; i < (mbs - 1) * bs; i++) {
      col[0] = i + bs;
      PetscCall(MatSetValues(C, 1, &i, 1, col, value, INSERT_VALUES));
      col[0] = i;
      row    = i + bs;
      PetscCall(MatSetValues(C, 1, &row, 1, col, value, INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));
  }
  {
    /* Check the symmetry of C because it will be converted to a sbaij matrix */
    Mat Ctrans;
    PetscCall(MatTranspose(C, MAT_INITIAL_MATRIX, &Ctrans));
    PetscCall(MatEqual(C, Ctrans, &flg));
    /*
    {
      PetscCall(MatAXPY(C,1.,Ctrans,DIFFERENT_NONZERO_PATTERN));
      flg  = PETSC_TRUE;
    }
*/
    PetscCall(MatSetOption(C, MAT_SYMMETRIC, flg));
    PetscCall(MatDestroy(&Ctrans));
  }
  PetscCall(MatIsSymmetric(C, 0.0, &issymmetric));
  PetscCall(MatViewFromOptions(C, NULL, "-view_mat"));

  /* convert C to other formats */
  for (i = 0; i < ntypes; i++) {
    PetscBool ismpisbaij, isseqsbaij;

    PetscCall(PetscStrcmp(type[i], MATMPISBAIJ, &ismpisbaij));
    PetscCall(PetscStrcmp(type[i], MATMPISBAIJ, &isseqsbaij));
    if (!issymmetric && (ismpisbaij || isseqsbaij)) continue;
    PetscCall(MatConvert(C, type[i], MAT_INITIAL_MATRIX, &A));
    PetscCall(MatMultEqual(A, C, 10, &equal));
    PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_ARG_NOTSAMETYPE, "Error in conversion from BAIJ to %s", type[i]);
    for (j = i + 1; j < ntypes; j++) {
      PetscCall(PetscStrcmp(type[j], MATMPISBAIJ, &ismpisbaij));
      PetscCall(PetscStrcmp(type[j], MATMPISBAIJ, &isseqsbaij));
      if (!issymmetric && (ismpisbaij || isseqsbaij)) continue;
      if (verbose > 0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, " \n[%d] test conversion between %s and %s\n", rank, type[i], type[j]));

      if (rank == 0 && verbose) printf("Convert %s A to %s B\n", type[i], type[j]);
      PetscCall(MatConvert(A, type[j], MAT_INITIAL_MATRIX, &B));
      /*
      if (j == 2) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF," A: %s\n",type[i]));
        PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
        PetscCall(PetscPrintf(PETSC_COMM_SELF," B: %s\n",type[j]));
        PetscCall(MatView(B,PETSC_VIEWER_STDOUT_WORLD));
      }
       */
      PetscCall(MatMultEqual(A, B, 10, &equal));
      PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_ARG_NOTSAMETYPE, "Error in conversion from %s to %s", type[i], type[j]);

      if (size == 1 || j != 2) { /* Matconvert from mpisbaij mat to other formats are not supported */
        if (rank == 0 && verbose) printf("Convert %s B to %s D\n", type[j], type[i]);
        PetscCall(MatConvert(B, type[i], MAT_INITIAL_MATRIX, &D));
        PetscCall(MatMultEqual(B, D, 10, &equal));
        PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_ARG_NOTSAMETYPE, "Error in conversion from %s to %s", type[j], type[i]);

        PetscCall(MatDestroy(&D));
      }
      PetscCall(MatDestroy(&B));
      PetscCall(MatDestroy(&D));
    }

    /* Test in-place convert */
    if (size == 1) { /* size > 1 is not working yet! */
      j = (i + 1) % ntypes;
      PetscCall(PetscStrcmp(type[j], MATMPISBAIJ, &ismpisbaij));
      PetscCall(PetscStrcmp(type[j], MATMPISBAIJ, &isseqsbaij));
      if (!issymmetric && (ismpisbaij || isseqsbaij)) continue;
      /* printf("[%d] i: %d, j: %d\n",rank,i,j); */
      PetscCall(MatConvert(A, type[j], MAT_INPLACE_MATRIX, &A));
    }

    PetscCall(MatDestroy(&A));
  }

  /* test BAIJ to MATIS */
  if (size > 1) {
    MatType ctype;

    PetscCall(MatGetType(C, &ctype));
    PetscCall(MatConvert(C, MATIS, MAT_INITIAL_MATRIX, &A));
    PetscCall(MatMultEqual(A, C, 10, &equal));
    PetscCall(MatViewFromOptions(A, NULL, "-view_conv"));
    if (!equal) {
      Mat C2;

      PetscCall(MatConvert(A, ctype, MAT_INITIAL_MATRIX, &C2));
      PetscCall(MatViewFromOptions(C2, NULL, "-view_conv_assembled"));
      PetscCall(MatAXPY(C2, -1., C, DIFFERENT_NONZERO_PATTERN));
      PetscCall(MatChop(C2, PETSC_SMALL));
      PetscCall(MatViewFromOptions(C2, NULL, "-view_err"));
      PetscCall(MatDestroy(&C2));
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error in conversion from BAIJ to MATIS");
    }
    PetscCall(MatConvert(C, MATIS, MAT_REUSE_MATRIX, &A));
    PetscCall(MatMultEqual(A, C, 10, &equal));
    PetscCall(MatViewFromOptions(A, NULL, "-view_conv"));
    if (!equal) {
      Mat C2;

      PetscCall(MatConvert(A, ctype, MAT_INITIAL_MATRIX, &C2));
      PetscCall(MatViewFromOptions(C2, NULL, "-view_conv_assembled"));
      PetscCall(MatAXPY(C2, -1., C, DIFFERENT_NONZERO_PATTERN));
      PetscCall(MatChop(C2, PETSC_SMALL));
      PetscCall(MatViewFromOptions(C2, NULL, "-view_err"));
      PetscCall(MatDestroy(&C2));
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error in conversion from BAIJ to MATIS");
    }
    PetscCall(MatDestroy(&A));
    PetscCall(MatDuplicate(C, MAT_COPY_VALUES, &A));
    PetscCall(MatConvert(A, MATIS, MAT_INPLACE_MATRIX, &A));
    PetscCall(MatViewFromOptions(A, NULL, "-view_conv"));
    PetscCall(MatMultEqual(A, C, 10, &equal));
    if (!equal) {
      Mat C2;

      PetscCall(MatViewFromOptions(A, NULL, "-view_conv"));
      PetscCall(MatConvert(A, ctype, MAT_INITIAL_MATRIX, &C2));
      PetscCall(MatViewFromOptions(C2, NULL, "-view_conv_assembled"));
      PetscCall(MatAXPY(C2, -1., C, DIFFERENT_NONZERO_PATTERN));
      PetscCall(MatChop(C2, PETSC_SMALL));
      PetscCall(MatViewFromOptions(C2, NULL, "-view_err"));
      PetscCall(MatDestroy(&C2));
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error in conversion from BAIJ to MATIS");
    }
    PetscCall(MatDestroy(&A));
  }
  PetscCall(MatDestroy(&C));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

   test:
      suffix: 2
      nsize: 3

   testset:
      requires: parmetis
      output_file: output/ex55_1.out
      nsize: 3
      args: -mat_is_disassemble_l2g_type nd -mat_partitioning_type parmetis
      test:
        suffix: matis_baij_parmetis_nd
      test:
        suffix: matis_aij_parmetis_nd
        args: -testseqaij
      test:
        requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
        suffix: matis_poisson1_parmetis_nd
        args: -f ${DATAFILESPATH}/matrices/poisson1

   testset:
      requires: ptscotch defined(PETSC_HAVE_SCOTCH_PARMETIS_V3_NODEND)
      output_file: output/ex55_1.out
      nsize: 4
      args: -mat_is_disassemble_l2g_type nd -mat_partitioning_type ptscotch
      test:
        suffix: matis_baij_ptscotch_nd
      test:
        suffix: matis_aij_ptscotch_nd
        args: -testseqaij
      test:
        requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
        suffix: matis_poisson1_ptscotch_nd
        args: -f ${DATAFILESPATH}/matrices/poisson1

TEST*/
