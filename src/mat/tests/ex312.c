static char help[] = "Tests MatDenseUpdateColumnLayout().\n\n";

#include <petscmat.h>

static PetscErrorCode SetLinearValues(Vec x)
{
  PetscInt rstart, rend;

  PetscFunctionBeginUser;
  PetscCall(VecGetOwnershipRange(x, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) PetscCall(VecSetValue(x, i, (PetscScalar)(i + 1), INSERT_VALUES));
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat         A, S;
  Vec         x, y, yref;
  PetscLayout clayout, cmap, scmap;
  PetscReal   norm;
  PetscInt    M, N, nloc, rstart, rend, cstart, cend, xstart, xend;
  PetscBool   same;
  PetscMPIInt rank, size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  M    = 2 * size + 1;
  N    = size * (size + 1) / 2;
  nloc = rank + 1;

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, M, N));
  PetscCall(MatSetType(A, MATDENSE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) {
    for (PetscInt j = 0; j < N; j++) PetscCall(MatSetValue(A, i, j, (PetscScalar)(1 + i + 2 * j), INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /* Cache a full-column submatrix using the original column layout. */
  PetscCall(MatDenseGetSubMatrix(A, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, &S));
  PetscCall(MatDenseRestoreSubMatrix(A, &S));

  /* Build the MPI dense matrix-vector scatter for the original column layout. */
  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(SetLinearValues(x));
  PetscCall(MatMult(A, x, y));
  PetscCall(VecDuplicate(y, &yref));
  PetscCall(VecCopy(y, yref));
  PetscCall(VecDestroy(&x));

  PetscCall(PetscLayoutCreateFromSizes(PETSC_COMM_WORLD, nloc, N, 1, &clayout));
  PetscCall(PetscLayoutGetRange(clayout, &cstart, &cend));
  PetscCall(MatDenseUpdateColumnLayout(A, clayout));
  PetscCall(MatGetLayouts(A, NULL, &cmap));
  PetscCheck(cmap == clayout, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Matrix did not adopt the new column layout");
  PetscCall(PetscLayoutDestroy(&clayout));

  PetscCall(MatCreateVecs(A, &x, NULL));
  PetscCall(VecGetOwnershipRange(x, &xstart, &xend));
  PetscCheck(xstart == cstart && xend == cend, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Right vector ownership range [%" PetscInt_FMT ",%" PetscInt_FMT ") does not match the new column layout [%" PetscInt_FMT ",%" PetscInt_FMT ")", xstart, xend, cstart, cend);
  PetscCall(SetLinearValues(x));

  PetscCall(MatDenseGetSubMatrix(A, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, &S));
  PetscCall(MatGetLayouts(S, NULL, &scmap));
  PetscCall(PetscLayoutCompare(scmap, cmap, &same));
  PetscCheck(same, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Submatrix did not adopt the new column layout");
  PetscCall(MatMult(S, x, y));
  PetscCall(MatDenseRestoreSubMatrix(A, &S));
  PetscCall(VecAXPY(y, -1.0, yref));
  PetscCall(VecNorm(y, NORM_INFINITY, &norm));
  PetscCheck(norm <= 100.0 * PETSC_MACHINE_EPSILON, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Submatrix MatMult() changed after updating the column layout, error %g", (double)norm);

  PetscCall(MatMult(A, x, y));
  PetscCall(VecAXPY(y, -1.0, yref));
  PetscCall(VecNorm(y, NORM_INFINITY, &norm));
  PetscCheck(norm <= 100.0 * PETSC_MACHINE_EPSILON, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "MatMult() changed after updating the column layout, error %g", (double)norm);

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&yref));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    nsize: {{1 2}}
    output_file: output/empty.out

    test:
      suffix: cpu

    test:
      requires: cuda
      suffix: cuda
      args: -mat_type densecuda

    test:
      requires: hip
      suffix: hip
      args: -mat_type densehip

TEST*/
