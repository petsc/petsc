
static char help[] = "Tests MatConvert(), MatLoad() for MATSCALAPACK interface.\n\n";
/*
 Example:
   mpiexec -n <np> ./ex244 -fA <A_data> -fB <B_data> -orig_mat_type <type> -orig_mat_type <mat_type>
*/

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat         A, Ae, B, Be;
  PetscViewer view;
  char        file[2][PETSC_MAX_PATH_LEN];
  PetscBool   flg, flgB, isScaLAPACK, isDense, isAij, isSbaij;
  PetscScalar one = 1.0;
  PetscMPIInt rank, size;
  PetscInt    M, N;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  /* Load PETSc matrices */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-fA", file[0], PETSC_MAX_PATH_LEN, NULL));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file[0], FILE_MODE_READ, &view));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetOptionsPrefix(A, "orig_"));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatLoad(A, view));
  PetscCall(PetscViewerDestroy(&view));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-fB", file[1], PETSC_MAX_PATH_LEN, &flgB));
  if (flgB) {
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file[1], FILE_MODE_READ, &view));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
    PetscCall(MatSetOptionsPrefix(B, "orig_"));
    PetscCall(MatSetType(B, MATAIJ));
    PetscCall(MatSetFromOptions(B));
    PetscCall(MatLoad(B, view));
    PetscCall(PetscViewerDestroy(&view));
  } else {
    /* Create matrix B = I */
    PetscInt rstart, rend, i;
    PetscCall(MatGetSize(A, &M, &N));
    PetscCall(MatGetOwnershipRange(A, &rstart, &rend));

    PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
    PetscCall(MatSetOptionsPrefix(B, "orig_"));
    PetscCall(MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, M, N));
    PetscCall(MatSetType(B, MATAIJ));
    PetscCall(MatSetFromOptions(B));
    PetscCall(MatSetUp(B));
    for (i = rstart; i < rend; i++) PetscCall(MatSetValues(B, 1, &i, 1, &i, &one, ADD_VALUES));
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  }

  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSCALAPACK, &isScaLAPACK));
  if (isScaLAPACK) {
    Ae      = A;
    Be      = B;
    isDense = isAij = isSbaij = PETSC_FALSE;
  } else { /* Convert AIJ/DENSE/SBAIJ matrices into ScaLAPACK matrices */
    if (size == 1) {
      PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQDENSE, &isDense));
      PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJ, &isAij));
      PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQSBAIJ, &isSbaij));
    } else {
      PetscCall(PetscObjectTypeCompare((PetscObject)A, MATMPIDENSE, &isDense));
      PetscCall(PetscObjectTypeCompare((PetscObject)A, MATMPIAIJ, &isAij));
      PetscCall(PetscObjectTypeCompare((PetscObject)A, MATMPISBAIJ, &isSbaij));
    }

    if (rank == 0) {
      if (isDense) {
        printf(" Convert DENSE matrices A and B into ScaLAPACK matrix... \n");
      } else if (isAij) {
        printf(" Convert AIJ matrices A and B into ScaLAPACK matrix... \n");
      } else if (isSbaij) {
        printf(" Convert SBAIJ matrices A and B into ScaLAPACK matrix... \n");
      } else SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Not supported yet");
    }
    PetscCall(MatConvert(A, MATSCALAPACK, MAT_INITIAL_MATRIX, &Ae));
    PetscCall(MatConvert(B, MATSCALAPACK, MAT_INITIAL_MATRIX, &Be));

    /* Test accuracy */
    PetscCall(MatMultEqual(A, Ae, 5, &flg));
    PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_ARG_NOTSAMETYPE, "A != A_scalapack.");
    PetscCall(MatMultEqual(B, Be, 5, &flg));
    PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_ARG_NOTSAMETYPE, "B != B_scalapack.");
  }

  if (!isScaLAPACK) {
    PetscCall(MatDestroy(&Ae));
    PetscCall(MatDestroy(&Be));

    /* Test MAT_REUSE_MATRIX which is only supported for inplace conversion */
    PetscCall(MatConvert(A, MATSCALAPACK, MAT_INPLACE_MATRIX, &A));
    //PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  }

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: scalapack

   test:
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_A_aij -fB ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_B_aij
      output_file: output/ex244.out

   test:
      suffix: 2
      nsize: 8
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_A_aij -fB ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_B_aij
      output_file: output/ex244.out

   test:
      suffix: 2_dense
      nsize: 8
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_A_aij -fB ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_B_aij -orig_mat_type dense
      output_file: output/ex244_dense.out

   test:
      suffix: 2_scalapack
      nsize: 8
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_A_aij -fB ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_B_aij -orig_mat_type scalapack
      output_file: output/ex244_scalapack.out

   test:
      suffix: 2_sbaij
      nsize: 8
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_A -fB ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_B -orig_mat_type sbaij
      output_file: output/ex244_sbaij.out

   test:
      suffix: complex
      requires: complex double datafilespath !defined(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/nimrod/small_112905
      output_file: output/ex244.out

   test:
      suffix: complex_2
      nsize: 4
      requires: complex double datafilespath !defined(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/nimrod/small_112905
      output_file: output/ex244.out

   test:
      suffix: dense
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_A_aij -fB ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_B_aij -orig_mat_type dense

   test:
      suffix: scalapack
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_A_aij -fB ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_B_aij -orig_mat_type scalapack

   test:
      suffix: sbaij
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -fA ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_A -fB ${DATAFILESPATH}/matrices/EigenProblems/Eigdftb/dftb_bin/graphene_xxs_B -orig_mat_type sbaij

TEST*/
