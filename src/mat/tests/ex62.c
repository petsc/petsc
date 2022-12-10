
static char help[] = "Test Matrix products for AIJ matrices\n\
Input arguments are:\n\
  -fA <input_file> -fB <input_file> -fC <input_file>: file to load\n\n";
/* Example of usage:
   ./ex62 -fA <A_binary> -fB <B_binary>
   mpiexec -n 3 ./ex62 -fA medium -fB medium
*/

#include <petscmat.h>

/*
     B = A - B
     norm = norm(B)
*/
PetscErrorCode MatNormDifference(Mat A, Mat B, PetscReal *norm)
{
  PetscFunctionBegin;
  PetscCall(MatAXPY(B, -1.0, A, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatNorm(B, NORM_FROBENIUS, norm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat          A, A_save, B, C, P, C1, R;
  PetscViewer  viewer;
  PetscMPIInt  size, rank;
  PetscInt     i, j, *idxn, PM, PN = PETSC_DECIDE, rstart, rend;
  PetscReal    norm;
  PetscRandom  rdm;
  char         file[2][PETSC_MAX_PATH_LEN] = {"", ""};
  PetscScalar *a, rval, alpha;
  PetscBool    Test_MatMatMult = PETSC_TRUE, Test_MatTrMat = PETSC_TRUE, Test_MatMatTr = PETSC_TRUE;
  PetscBool    Test_MatPtAP = PETSC_TRUE, Test_MatRARt = PETSC_TRUE, flg, seqaij, flgA, flgB;
  MatInfo      info;
  PetscInt     nzp = 5; /* num of nonzeros in each row of P */
  MatType      mattype;
  const char  *deft = MATAIJ;
  char         A_mattype[256], B_mattype[256];
  PetscInt     mcheck = 10;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /*  Load the matrices A_save and B */
  PetscOptionsBegin(PETSC_COMM_WORLD, "", "", "");
  PetscCall(PetscOptionsBool("-test_rart", "Test MatRARt", "", Test_MatRARt, &Test_MatRARt, NULL));
  PetscCall(PetscOptionsInt("-PN", "Number of columns of P", "", PN, &PN, NULL));
  PetscCall(PetscOptionsInt("-mcheck", "Number of matmult checks", "", mcheck, &mcheck, NULL));
  PetscCall(PetscOptionsString("-fA", "Path for matrix A", "", file[0], file[0], sizeof(file[0]), &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Must indicate a file name for matrix A with the -fA option.");
  PetscCall(PetscOptionsString("-fB", "Path for matrix B", "", file[1], file[1], sizeof(file[1]), &flg));
  PetscCall(PetscOptionsFList("-A_mat_type", "Matrix type", "MatSetType", MatList, deft, A_mattype, 256, &flgA));
  PetscCall(PetscOptionsFList("-B_mat_type", "Matrix type", "MatSetType", MatList, deft, B_mattype, 256, &flgB));
  PetscOptionsEnd();

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file[0], FILE_MODE_READ, &viewer));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A_save));
  PetscCall(MatLoad(A_save, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  if (flg) {
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file[1], FILE_MODE_READ, &viewer));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
    PetscCall(MatLoad(B, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  } else {
    PetscCall(PetscObjectReference((PetscObject)A_save));
    B = A_save;
  }

  if (flgA) PetscCall(MatConvert(A_save, A_mattype, MAT_INPLACE_MATRIX, &A_save));
  if (flgB) PetscCall(MatConvert(B, B_mattype, MAT_INPLACE_MATRIX, &B));
  PetscCall(MatSetFromOptions(A_save));
  PetscCall(MatSetFromOptions(B));

  PetscCall(MatGetType(B, &mattype));

  PetscCall(PetscMalloc(nzp * (sizeof(PetscInt) + sizeof(PetscScalar)), &idxn));
  a = (PetscScalar *)(idxn + nzp);

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));

  /* 1) MatMatMult() */
  /* ----------------*/
  if (Test_MatMatMult) {
    PetscCall(MatDuplicate(A_save, MAT_COPY_VALUES, &A));

    /* (1.1) Test developer API */
    PetscCall(MatProductCreate(A, B, NULL, &C));
    PetscCall(MatSetOptionsPrefix(C, "AB_"));
    PetscCall(MatProductSetType(C, MATPRODUCT_AB));
    PetscCall(MatProductSetAlgorithm(C, MATPRODUCTALGORITHMDEFAULT));
    PetscCall(MatProductSetFill(C, PETSC_DEFAULT));
    PetscCall(MatProductSetFromOptions(C));
    /* we can inquire about MATOP_PRODUCTSYMBOLIC even if the destination matrix type has not been set yet */
    PetscCall(MatHasOperation(C, MATOP_PRODUCTSYMBOLIC, &flg));
    PetscCall(MatProductSymbolic(C));
    PetscCall(MatProductNumeric(C));
    PetscCall(MatMatMultEqual(A, B, C, mcheck, &flg));
    PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error in C=A*B");

    /* Test reuse symbolic C */
    alpha = 0.9;
    PetscCall(MatScale(A, alpha));
    PetscCall(MatProductNumeric(C));

    PetscCall(MatMatMultEqual(A, B, C, mcheck, &flg));
    PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error in C=A*B");
    PetscCall(MatDestroy(&C));

    /* (1.2) Test user driver */
    PetscCall(MatMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C));

    /* Test MAT_REUSE_MATRIX - reuse symbolic C */
    alpha = 1.0;
    for (i = 0; i < 2; i++) {
      alpha -= 0.1;
      PetscCall(MatScale(A, alpha));
      PetscCall(MatMatMult(A, B, MAT_REUSE_MATRIX, PETSC_DEFAULT, &C));
    }
    PetscCall(MatMatMultEqual(A, B, C, mcheck, &flg));
    PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error: MatMatMult()");
    PetscCall(MatDestroy(&A));

    /* Test MatProductClear() */
    PetscCall(MatProductClear(C));
    PetscCall(MatDestroy(&C));

    /* Test MatMatMult() for dense and aij matrices */
    PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &flg, MATSEQAIJ, MATMPIAIJ, ""));
    if (flg) {
      PetscCall(MatConvert(A_save, MATDENSE, MAT_INITIAL_MATRIX, &A));
      PetscCall(MatMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C));
      PetscCall(MatDestroy(&C));
      PetscCall(MatDestroy(&A));
    }
  }

  /* Create P and R = P^T  */
  /* --------------------- */
  PetscCall(MatGetSize(B, &PM, NULL));
  if (PN < 0) PN = PM / 2;
  PetscCall(MatCreate(PETSC_COMM_WORLD, &P));
  PetscCall(MatSetSizes(P, PETSC_DECIDE, PETSC_DECIDE, PM, PN));
  PetscCall(MatSetType(P, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(P, nzp, NULL));
  PetscCall(MatMPIAIJSetPreallocation(P, nzp, NULL, nzp, NULL));
  PetscCall(MatGetOwnershipRange(P, &rstart, &rend));
  for (i = 0; i < nzp; i++) PetscCall(PetscRandomGetValue(rdm, &a[i]));
  for (i = rstart; i < rend; i++) {
    for (j = 0; j < nzp; j++) {
      PetscCall(PetscRandomGetValue(rdm, &rval));
      idxn[j] = (PetscInt)(PetscRealPart(rval) * PN);
    }
    PetscCall(MatSetValues(P, 1, &i, nzp, idxn, a, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));

  PetscCall(MatTranspose(P, MAT_INITIAL_MATRIX, &R));
  PetscCall(MatConvert(P, mattype, MAT_INPLACE_MATRIX, &P));
  PetscCall(MatConvert(R, mattype, MAT_INPLACE_MATRIX, &R));
  PetscCall(MatSetFromOptions(P));
  PetscCall(MatSetFromOptions(R));

  /* 2) MatTransposeMatMult() */
  /* ------------------------ */
  if (Test_MatTrMat) {
    /* (2.1) Test developer driver C = P^T*B */
    PetscCall(MatProductCreate(P, B, NULL, &C));
    PetscCall(MatSetOptionsPrefix(C, "AtB_"));
    PetscCall(MatProductSetType(C, MATPRODUCT_AtB));
    PetscCall(MatProductSetAlgorithm(C, MATPRODUCTALGORITHMDEFAULT));
    PetscCall(MatProductSetFill(C, PETSC_DEFAULT));
    PetscCall(MatProductSetFromOptions(C));
    PetscCall(MatHasOperation(C, MATOP_PRODUCTSYMBOLIC, &flg));
    if (flg) {                                                 /* run tests if supported */
      PetscCall(MatProductSymbolic(C));                        /* equivalent to MatSetUp() */
      PetscCall(MatSetOption(C, MAT_USE_INODES, PETSC_FALSE)); /* illustrate how to call MatSetOption() */
      PetscCall(MatProductNumeric(C));
      PetscCall(MatProductNumeric(C)); /* test reuse symbolic C */

      PetscCall(MatTransposeMatMultEqual(P, B, C, mcheck, &flg));
      PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Error: developer driver C = P^T*B");
      PetscCall(MatDestroy(&C));

      /* (2.2) Test user driver C = P^T*B */
      PetscCall(MatTransposeMatMult(P, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C));
      PetscCall(MatTransposeMatMult(P, B, MAT_REUSE_MATRIX, PETSC_DEFAULT, &C));
      PetscCall(MatGetInfo(C, MAT_GLOBAL_SUM, &info));
      PetscCall(MatProductClear(C));

      /* Compare P^T*B and R*B */
      PetscCall(MatMatMult(R, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C1));
      PetscCall(MatNormDifference(C, C1, &norm));
      PetscCheck(norm <= PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Error in MatTransposeMatMult(): %g", (double)norm);
      PetscCall(MatDestroy(&C1));

      /* Test MatDuplicate() of C=P^T*B */
      PetscCall(MatDuplicate(C, MAT_COPY_VALUES, &C1));
      PetscCall(MatDestroy(&C1));
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatTransposeMatMult not supported\n"));
    }
    PetscCall(MatDestroy(&C));
  }

  /* 3) MatMatTransposeMult() */
  /* ------------------------ */
  if (Test_MatMatTr) {
    /* C = B*R^T */
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATSEQAIJ, &seqaij));
    if (seqaij) {
      PetscCall(MatMatTransposeMult(B, R, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C));
      PetscCall(MatSetOptionsPrefix(C, "ABt_")); /* enable '-ABt_' for matrix C */
      PetscCall(MatGetInfo(C, MAT_GLOBAL_SUM, &info));

      /* Test MAT_REUSE_MATRIX - reuse symbolic C */
      PetscCall(MatMatTransposeMult(B, R, MAT_REUSE_MATRIX, PETSC_DEFAULT, &C));

      /* Check */
      PetscCall(MatMatMult(B, P, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C1));
      PetscCall(MatNormDifference(C, C1, &norm));
      PetscCheck(norm <= PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Error in MatMatTransposeMult() %g", (double)norm);
      PetscCall(MatDestroy(&C1));
      PetscCall(MatDestroy(&C));
    }
  }

  /* 4) Test MatPtAP() */
  /*-------------------*/
  if (Test_MatPtAP) {
    PetscCall(MatDuplicate(A_save, MAT_COPY_VALUES, &A));

    /* (4.1) Test developer API */
    PetscCall(MatProductCreate(A, P, NULL, &C));
    PetscCall(MatSetOptionsPrefix(C, "PtAP_"));
    PetscCall(MatProductSetType(C, MATPRODUCT_PtAP));
    PetscCall(MatProductSetAlgorithm(C, MATPRODUCTALGORITHMDEFAULT));
    PetscCall(MatProductSetFill(C, PETSC_DEFAULT));
    PetscCall(MatProductSetFromOptions(C));
    PetscCall(MatProductSymbolic(C));
    PetscCall(MatProductNumeric(C));
    PetscCall(MatPtAPMultEqual(A, P, C, mcheck, &flg));
    PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Error in MatProduct_PtAP");
    PetscCall(MatProductNumeric(C)); /* reuse symbolic C */

    PetscCall(MatPtAPMultEqual(A, P, C, mcheck, &flg));
    PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Error in MatProduct_PtAP");
    PetscCall(MatDestroy(&C));

    /* (4.2) Test user driver */
    PetscCall(MatPtAP(A, P, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C));

    /* Test MAT_REUSE_MATRIX - reuse symbolic C */
    alpha = 1.0;
    for (i = 0; i < 2; i++) {
      alpha -= 0.1;
      PetscCall(MatScale(A, alpha));
      PetscCall(MatPtAP(A, P, MAT_REUSE_MATRIX, PETSC_DEFAULT, &C));
    }
    PetscCall(MatPtAPMultEqual(A, P, C, mcheck, &flg));
    PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Error in MatPtAP");

    /* 5) Test MatRARt() */
    /* ----------------- */
    if (Test_MatRARt) {
      Mat RARt;

      /* (5.1) Test developer driver RARt = R*A*Rt */
      PetscCall(MatProductCreate(A, R, NULL, &RARt));
      PetscCall(MatSetOptionsPrefix(RARt, "RARt_"));
      PetscCall(MatProductSetType(RARt, MATPRODUCT_RARt));
      PetscCall(MatProductSetAlgorithm(RARt, MATPRODUCTALGORITHMDEFAULT));
      PetscCall(MatProductSetFill(RARt, PETSC_DEFAULT));
      PetscCall(MatProductSetFromOptions(RARt));
      PetscCall(MatHasOperation(RARt, MATOP_PRODUCTSYMBOLIC, &flg));
      if (flg) {
        PetscCall(MatProductSymbolic(RARt));                        /* equivalent to MatSetUp() */
        PetscCall(MatSetOption(RARt, MAT_USE_INODES, PETSC_FALSE)); /* illustrate how to call MatSetOption() */
        PetscCall(MatProductNumeric(RARt));
        PetscCall(MatProductNumeric(RARt)); /* test reuse symbolic RARt */
        PetscCall(MatDestroy(&RARt));

        /* (2.2) Test user driver RARt = R*A*Rt */
        PetscCall(MatRARt(A, R, MAT_INITIAL_MATRIX, 2.0, &RARt));
        PetscCall(MatRARt(A, R, MAT_REUSE_MATRIX, 2.0, &RARt));

        PetscCall(MatNormDifference(C, RARt, &norm));
        PetscCheck(norm <= PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "|PtAP - RARt| = %g", (double)norm);
      } else {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatRARt not supported\n"));
      }
      PetscCall(MatDestroy(&RARt));
    }

    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&C));
  }

  /* Destroy objects */
  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(PetscFree(idxn));

  PetscCall(MatDestroy(&A_save));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&P));
  PetscCall(MatDestroy(&R));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
   test:
     suffix: 1
     requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
     args: -fA ${DATAFILESPATH}/matrices/medium -fB ${DATAFILESPATH}/matrices/medium
     output_file: output/ex62_1.out

   test:
     suffix: 2_ab_scalable
     requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
     args: -fA ${DATAFILESPATH}/matrices/medium -fB ${DATAFILESPATH}/matrices/medium -AB_mat_product_algorithm scalable -matmatmult_via scalable -AtB_mat_product_algorithm outerproduct -mattransposematmult_via outerproduct
     output_file: output/ex62_1.out

   test:
     suffix: 3_ab_scalable_fast
     requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
     args: -fA ${DATAFILESPATH}/matrices/medium -fB ${DATAFILESPATH}/matrices/medium -AB_mat_product_algorithm scalable_fast -matmatmult_via scalable_fast -matmattransmult_via color
     output_file: output/ex62_1.out

   test:
     suffix: 4_ab_heap
     requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
     args: -fA ${DATAFILESPATH}/matrices/medium -fB ${DATAFILESPATH}/matrices/medium -AB_mat_product_algorithm heap -matmatmult_via heap -PtAP_mat_product_algorithm rap -matptap_via rap
     output_file: output/ex62_1.out

   test:
     suffix: 5_ab_btheap
     requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
     args: -fA ${DATAFILESPATH}/matrices/medium -fB ${DATAFILESPATH}/matrices/medium -AB_mat_product_algorithm btheap -matmatmult_via btheap -matrart_via r*art
     output_file: output/ex62_1.out

   test:
     suffix: 6_ab_llcondensed
     requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
     args: -fA ${DATAFILESPATH}/matrices/medium -fB ${DATAFILESPATH}/matrices/medium -AB_mat_product_algorithm llcondensed -matmatmult_via llcondensed -matrart_via coloring_rart
     output_file: output/ex62_1.out

   test:
     suffix: 7_ab_rowmerge
     requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
     args: -fA ${DATAFILESPATH}/matrices/medium -fB ${DATAFILESPATH}/matrices/medium -AB_mat_product_algorithm rowmerge -matmatmult_via rowmerge
     output_file: output/ex62_1.out

   test:
     suffix: 8_ab_hypre
     requires: hypre datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
     args: -fA ${DATAFILESPATH}/matrices/medium -fB ${DATAFILESPATH}/matrices/medium -AB_mat_product_algorithm hypre -matmatmult_via hypre -PtAP_mat_product_algorithm hypre -matptap_via hypre
     output_file: output/ex62_1.out

   test:
     suffix: hypre_medium
     nsize: {{1 3}}
     requires: hypre datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
     args: -fA ${DATAFILESPATH}/matrices/medium -fB ${DATAFILESPATH}/matrices/medium -A_mat_type hypre -B_mat_type hypre -test_rart 0
     output_file: output/ex62_hypre.out

   test:
     suffix: hypre_tiny
     nsize: {{1 3}}
     requires: hypre !complex double !defined(PETSC_USE_64BIT_INDICES)
     args: -fA ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system -fB ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system -A_mat_type hypre -B_mat_type hypre -test_rart 0
     output_file: output/ex62_hypre.out

   test:
     suffix: 9_mkl
     TODO: broken MatScale?
     requires: mkl_sparse datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
     args: -fA ${DATAFILESPATH}/matrices/medium -fB ${DATAFILESPATH}/matrices/medium -A_mat_type aijmkl -B_mat_type aijmkl
     output_file: output/ex62_1.out

   test:
     suffix: 10
     requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
     nsize: 3
     args: -fA ${DATAFILESPATH}/matrices/medium -fB ${DATAFILESPATH}/matrices/medium
     output_file: output/ex62_1.out

   test:
     suffix: 10_backend
     requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
     nsize: 3
     args: -fA ${DATAFILESPATH}/matrices/medium -AB_mat_product_algorithm backend -matmatmult_via backend -AtB_mat_product_algorithm backend -mattransposematmult_via backend -PtAP_mat_product_algorithm backend -matptap_via backend
     output_file: output/ex62_1.out

   test:
     suffix: 11_ab_scalable
     requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
     nsize: 3
     args: -fA ${DATAFILESPATH}/matrices/medium -fB ${DATAFILESPATH}/matrices/medium -AB_mat_product_algorithm scalable -matmatmult_via scalable -AtB_mat_product_algorithm scalable -mattransposematmult_via scalable
     output_file: output/ex62_1.out

   test:
     suffix: 12_ab_seqmpi
     requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
     nsize: 3
     args: -fA ${DATAFILESPATH}/matrices/medium -fB ${DATAFILESPATH}/matrices/medium -AB_mat_product_algorithm seqmpi -matmatmult_via seqmpi -AtB_mat_product_algorithm at*b -mattransposematmult_via at*b
     output_file: output/ex62_1.out

   test:
     suffix: 13_ab_hypre
     requires: hypre datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
     nsize: 3
     args: -fA ${DATAFILESPATH}/matrices/medium -fB ${DATAFILESPATH}/matrices/medium -AB_mat_product_algorithm hypre -matmatmult_via hypre -PtAP_mat_product_algorithm hypre -matptap_via hypre
     output_file: output/ex62_1.out

   test:
     suffix: 14_seqaij
     requires: !complex double !defined(PETSC_USE_64BIT_INDICES)
     args: -fA ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system -fB ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system
     output_file: output/ex62_1.out

   test:
     suffix: 14_seqaijcusparse
     requires: cuda !complex double !defined(PETSC_USE_64BIT_INDICES)
     args: -A_mat_type aijcusparse -B_mat_type aijcusparse -mat_form_explicit_transpose -fA ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system -fB ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system
     output_file: output/ex62_1.out

   test:
     suffix: 14_seqaijcusparse_cpu
     requires: cuda !complex double !defined(PETSC_USE_64BIT_INDICES)
     args: -A_mat_type aijcusparse -B_mat_type aijcusparse -mat_form_explicit_transpose -fA ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system -fB ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system -AB_mat_product_algorithm_backend_cpu -matmatmult_backend_cpu -PtAP_mat_product_algorithm_backend_cpu -matptap_backend_cpu -RARt_mat_product_algorithm_backend_cpu -matrart_backend_cpu
     output_file: output/ex62_1.out

   test:
     suffix: 14_mpiaijcusparse_seq
     nsize: 1
     requires: cuda !complex double !defined(PETSC_USE_64BIT_INDICES)
     args: -A_mat_type mpiaijcusparse -B_mat_type mpiaijcusparse -mat_form_explicit_transpose -fA ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system -fB ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system
     output_file: output/ex62_1.out

   test:
     suffix: 14_mpiaijcusparse_seq_cpu
     nsize: 1
     requires: cuda !complex double !defined(PETSC_USE_64BIT_INDICES)
     args: -A_mat_type mpiaijcusparse -B_mat_type mpiaijcusparse -mat_form_explicit_transpose -fA ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system -fB ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system -AB_mat_product_algorithm_backend_cpu -matmatmult_backend_cpu -PtAP_mat_product_algorithm_backend_cpu -matptap_backend_cpu -test_rart 0
     output_file: output/ex62_1.out

   test:
     suffix: 14_mpiaijcusparse
     nsize: 3
     requires: cuda !complex double !defined(PETSC_USE_64BIT_INDICES)
     args: -A_mat_type mpiaijcusparse -B_mat_type mpiaijcusparse -mat_form_explicit_transpose -fA ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system -fB ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system
     output_file: output/ex62_1.out

   test:
     suffix: 14_mpiaijcusparse_cpu
     nsize: 3
     requires: cuda !complex double !defined(PETSC_USE_64BIT_INDICES)
     args: -A_mat_type mpiaijcusparse -B_mat_type mpiaijcusparse -mat_form_explicit_transpose -fA ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system -fB ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system -AB_mat_product_algorithm_backend_cpu -matmatmult_backend_cpu -PtAP_mat_product_algorithm_backend_cpu -matptap_backend_cpu -test_rart 0
     output_file: output/ex62_1.out

   test:
     nsize: {{1 3}}
     suffix: 14_aijkokkos
     requires: kokkos_kernels !complex double !defined(PETSC_USE_64BIT_INDICES)
     args: -A_mat_type aijkokkos -B_mat_type aijkokkos -fA ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system -fB ${wPETSC_DIR}/share/petsc/datafiles/matrices/tiny_system
     output_file: output/ex62_1.out

   # these tests use matrices with many zero rows
   test:
     suffix: 15_seqaijcusparse
     requires: cuda !complex double !defined(PETSC_USE_64BIT_INDICES) datafilespath
     args: -A_mat_type aijcusparse -mat_form_explicit_transpose -fA ${DATAFILESPATH}/matrices/matmatmult/A4.BGriffith
     output_file: output/ex62_1.out

   test:
     suffix: 15_mpiaijcusparse_seq
     nsize: 1
     requires: cuda !complex double !defined(PETSC_USE_64BIT_INDICES) datafilespath
     args: -A_mat_type mpiaijcusparse -mat_form_explicit_transpose -fA ${DATAFILESPATH}/matrices/matmatmult/A4.BGriffith
     output_file: output/ex62_1.out

   test:
     nsize: 3
     suffix: 15_mpiaijcusparse
     requires: cuda !complex double !defined(PETSC_USE_64BIT_INDICES) datafilespath
     args: -A_mat_type mpiaijcusparse -mat_form_explicit_transpose -fA ${DATAFILESPATH}/matrices/matmatmult/A4.BGriffith
     output_file: output/ex62_1.out

TEST*/
