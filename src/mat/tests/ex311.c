static const char help[] = "Test MatDiagonalScale() on dense matrices with scaling Vecs of any type\n\n";

// Contributed by: Steven Dargaville

#include <petscmat.h>

/* The scaling Vecs take their type from -vec_type, independently of the Mat type set
   with -mat_type. Scaling is verified against a duplicate matrix scaled with VECSTANDARD
   Vecs holding the same values: for a device matrix both scalings execute the same kernel
   on identical data, so MatEqual() is exact, and no cross-type Vec or Mat copies are
   needed */
int main(int argc, char **args)
{
  Mat       A, B;
  Vec       l, r, lstd, rstd;
  PetscInt  m = 5, n = 4, mloc, nloc, rstart, rend;
  PetscBool equal = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, n));
  PetscCall(MatSetType(A, MATDENSE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetRandom(A, NULL));
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  PetscCall(MatGetLocalSize(A, &mloc, &nloc));

  // l and lstd match A's row layout, r and rstd its column layout
  PetscCall(VecCreate(PETSC_COMM_WORLD, &l));
  PetscCall(VecSetSizes(l, mloc, m));
  PetscCall(VecSetFromOptions(l));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &lstd));
  PetscCall(VecSetSizes(lstd, mloc, m));
  PetscCall(VecSetType(lstd, VECSTANDARD));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &r));
  PetscCall(VecSetSizes(r, nloc, n));
  PetscCall(VecSetFromOptions(r));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &rstd));
  PetscCall(VecSetSizes(rstd, nloc, n));
  PetscCall(VecSetType(rstd, VECSTANDARD));

  PetscCall(VecGetOwnershipRange(l, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) {
    PetscCall(VecSetValue(l, i, (PetscScalar)(i + 2), INSERT_VALUES));
    PetscCall(VecSetValue(lstd, i, (PetscScalar)(i + 2), INSERT_VALUES));
  }
  PetscCall(VecGetOwnershipRange(r, &rstart, &rend));
  for (PetscInt j = rstart; j < rend; j++) {
    PetscCall(VecSetValue(r, j, (PetscScalar)(j + 3), INSERT_VALUES));
    PetscCall(VecSetValue(rstd, j, (PetscScalar)(j + 3), INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(l));
  PetscCall(VecAssemblyEnd(l));
  PetscCall(VecAssemblyBegin(lstd));
  PetscCall(VecAssemblyEnd(lstd));
  PetscCall(VecAssemblyBegin(r));
  PetscCall(VecAssemblyEnd(r));
  PetscCall(VecAssemblyBegin(rstd));
  PetscCall(VecAssemblyEnd(rstd));

  // left scaling only
  PetscCall(MatDiagonalScale(A, l, NULL));
  PetscCall(MatDiagonalScale(B, lstd, NULL));
  PetscCall(MatEqual(A, B, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Left scaling gives the wrong result");

  // right scaling on top of the left scaling
  PetscCall(MatDiagonalScale(A, NULL, r));
  PetscCall(MatDiagonalScale(B, NULL, rstd));
  PetscCall(MatEqual(A, B, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Right scaling gives the wrong result");

  // both sides in one call
  PetscCall(MatDiagonalScale(A, l, r));
  PetscCall(MatDiagonalScale(B, lstd, rstd));
  PetscCall(MatEqual(A, B, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Two-sided scaling gives the wrong result");

  PetscCall(VecDestroy(&l));
  PetscCall(VecDestroy(&lstd));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&rstd));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: cpu
    nsize: {{1 2}}
    output_file: output/empty.out

  test:
    suffix: kokkos
    nsize: {{1 2}}
    requires: kokkos_kernels
    args: -vec_type kokkos
    output_file: output/empty.out

  test:
    suffix: cuda
    nsize: {{1 2}}
    requires: cuda
    args: -mat_type densecuda -vec_type {{cuda standard}}
    output_file: output/empty.out

  test:
    suffix: densecuda_vec_kokkos
    nsize: {{1 2}}
    requires: cuda kokkos_kernels
    args: -mat_type densecuda -vec_type kokkos
    output_file: output/empty.out

  test:
    suffix: hip
    nsize: {{1 2}}
    requires: hip
    args: -mat_type densehip -vec_type {{hip standard}}
    output_file: output/empty.out

  test:
    suffix: densehip_vec_kokkos
    nsize: {{1 2}}
    requires: hip kokkos_kernels
    args: -mat_type densehip -vec_type kokkos
    output_file: output/empty.out

TEST*/
