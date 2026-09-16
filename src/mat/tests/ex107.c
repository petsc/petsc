static char help[] = "Test MatCreate() with MAT_STRUCTURE_ONLY.\n\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat         mat;
  PetscInt    m = 7, n, nlocal, i, j, rstart, rend, bs;
  PetscMPIInt size;
  PetscScalar v;
  PetscBool   struct_only = PETSC_TRUE, explicit_preallocation = PETSC_FALSE, ismpiaij, ismpibaij, ismpisbaij;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_COMMON));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-struct_only", &struct_only, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-explicit_preallocation", &explicit_preallocation, NULL));
  n      = m;
  nlocal = PETSC_DECIDE;
  PetscCall(PetscSplitOwnership(PETSC_COMM_WORLD, &nlocal, &n));

  /* ------- Assemble matrix, test MatValid() --------- */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &mat));
  PetscCall(MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, m, n));
  PetscCall(MatSetFromOptions(mat));
  PetscCall(PetscObjectTypeCompare((PetscObject)mat, MATMPIAIJ, &ismpiaij));
  PetscCall(PetscObjectTypeCompare((PetscObject)mat, MATMPIBAIJ, &ismpibaij));
  PetscCall(PetscObjectTypeCompare((PetscObject)mat, MATMPISBAIJ, &ismpisbaij));
  if (struct_only) PetscCall(MatSetOption(mat, MAT_STRUCTURE_ONLY, PETSC_TRUE));
  if (explicit_preallocation) {
    PetscCall(MatGetBlockSize(mat, &bs));
    if (ismpiaij) PetscCall(MatMPIAIJSetPreallocation(mat, nlocal, NULL, n - nlocal, NULL));
    else if (ismpibaij) PetscCall(MatMPIBAIJSetPreallocation(mat, bs, nlocal / bs, NULL, (n - nlocal) / bs, NULL));
    else if (ismpisbaij) PetscCall(MatMPISBAIJSetPreallocation(mat, bs, nlocal / bs, NULL, (n - nlocal) / bs, NULL));
    else {
      PetscCall(MatSeqAIJSetPreallocation(mat, n, NULL));
      PetscCall(MatSeqBAIJSetPreallocation(mat, bs, n / bs, NULL));
      PetscCall(MatSeqSBAIJSetPreallocation(mat, bs, n / bs, NULL));
    }
  } else PetscCall(MatSetUp(mat));
  PetscCall(MatGetOwnershipRange(mat, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    for (j = 0; j < n; j++) {
      v = 10.0 * i + j;
      PetscCall(MatSetValues(mat, 1, &i, 1, &j, &v, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  if (size == 1) PetscCall(MatView(mat, PETSC_VIEWER_STDOUT_WORLD));

  /* Free data structures */
  PetscCall(MatDestroy(&mat));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      output_file: output/ex107.out

   test:
      suffix: 2
      args: -mat_type {{baij sbaij}separate output} -mat_block_size 2 -m 10

   testset:
      nsize: 2
      output_file: output/empty.out
      test:
         suffix: aij
         args: -explicit_preallocation {{true false}shared output}
      test:
         suffix: block
         args: -mat_type {{baij sbaij}shared output} -mat_block_size 2 -m 8 -explicit_preallocation {{true false}shared output}

TEST*/
