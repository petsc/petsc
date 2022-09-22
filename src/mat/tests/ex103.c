static char help[] = "Test MatSetValues() by converting MATDENSE to MATELEMENTAL. \n\
Modified from the code contributed by Yaning Liu @lbl.gov \n\n";
/*
 Example:
   mpiexec -n <np> ./ex103
   mpiexec -n <np> ./ex103 -mat_type elemental -mat_view
   mpiexec -n <np> ./ex103 -mat_type aij
*/

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat             A, A_elemental;
  PetscInt        i, j, M = 10, N = 5, nrows, ncols;
  PetscMPIInt     rank, size;
  IS              isrows, iscols;
  const PetscInt *rows, *cols;
  PetscScalar    *v;
  MatType         type;
  PetscBool       isDense, isAIJ, flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* Creat a matrix */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-M", &M, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, M, N));
  PetscCall(MatSetType(A, MATDENSE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  /* Set local matrix entries */
  PetscCall(MatGetOwnershipIS(A, &isrows, &iscols));
  PetscCall(ISGetLocalSize(isrows, &nrows));
  PetscCall(ISGetIndices(isrows, &rows));
  PetscCall(ISGetLocalSize(iscols, &ncols));
  PetscCall(ISGetIndices(iscols, &cols));
  PetscCall(PetscMalloc1(nrows * ncols, &v));

  for (i = 0; i < nrows; i++) {
    for (j = 0; j < ncols; j++) {
      if (size == 1) {
        v[i * ncols + j] = (PetscScalar)(i + j);
      } else {
        v[i * ncols + j] = (PetscScalar)rank + j * 0.1;
      }
    }
  }
  PetscCall(MatSetValues(A, nrows, rows, ncols, cols, v, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  //PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%" PetscInt_FMT "] local nrows %" PetscInt_FMT ", ncols %" PetscInt_FMT "\n",rank,nrows,ncols));
  //PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));

  /* Test MatSetValues() by converting A to A_elemental */
  PetscCall(MatGetType(A, &type));
  if (size == 1) {
    PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQDENSE, &isDense));
    PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJ, &isAIJ));
  } else {
    PetscCall(PetscObjectTypeCompare((PetscObject)A, MATMPIDENSE, &isDense));
    PetscCall(PetscObjectTypeCompare((PetscObject)A, MATMPIAIJ, &isAIJ));
  }

  if (isDense || isAIJ) {
    Mat Aexplicit;
    PetscCall(MatConvert(A, MATELEMENTAL, MAT_INITIAL_MATRIX, &A_elemental));
    PetscCall(MatComputeOperator(A_elemental, isAIJ ? MATAIJ : MATDENSE, &Aexplicit));
    PetscCall(MatMultEqual(Aexplicit, A_elemental, 5, &flg));
    PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Aexplicit != A_elemental.");
    PetscCall(MatDestroy(&Aexplicit));

    /* Test MAT_REUSE_MATRIX which is only supported for inplace conversion */
    PetscCall(MatConvert(A, MATELEMENTAL, MAT_INPLACE_MATRIX, &A));
    PetscCall(MatMultEqual(A_elemental, A, 5, &flg));
    PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "A_elemental != A.");
    PetscCall(MatDestroy(&A_elemental));
  }

  PetscCall(ISRestoreIndices(isrows, &rows));
  PetscCall(ISRestoreIndices(iscols, &cols));
  PetscCall(ISDestroy(&isrows));
  PetscCall(ISDestroy(&iscols));
  PetscCall(PetscFree(v));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: elemental

   test:
      nsize: 6

   test:
      suffix: 2
      nsize: 6
      args: -mat_type aij
      output_file: output/ex103_1.out

   test:
      suffix: 3
      nsize: 6
      args: -mat_type elemental
      output_file: output/ex103_1.out

TEST*/
