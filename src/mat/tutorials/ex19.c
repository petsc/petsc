const char help[] = "Test VecCreateMatDense()\n\n";

#include <petscdevice_cuda.h>
#include <petscmat.h>
#include <petscconf.h>
#include <assert.h>

int main(int argc, char **args)
{
  Mat      A;
  Vec      X;
  VecType  vtype;
  PetscInt n = 20, lda = PETSC_DECIDE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Creating Mat from Vec type example", NULL);
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-lda", &lda, NULL));
  PetscOptionsEnd();
  if (lda > 0) lda += n;

  PetscCall(VecCreate(PETSC_COMM_WORLD, &X));
  PetscCall(VecSetSizes(X, n, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(X));
  PetscCall(VecSetUp(X));
  PetscCall(VecGetType(X, &vtype));

  PetscCall(MatCreateDenseFromVecType(PETSC_COMM_WORLD, vtype, n, n, PETSC_DECIDE, PETSC_DECIDE, lda, NULL, &A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscMemType       X_memtype, A_memtype;
  const PetscScalar *array;
  PetscCall(VecGetArrayReadAndMemType(X, &array, &X_memtype));
  PetscCall(VecRestoreArrayReadAndMemType(X, &array));
  PetscCall(MatDenseGetArrayReadAndMemType(A, &array, &A_memtype));
  PetscCall(MatDenseRestoreArrayReadAndMemType(A, &array));
  PetscAssert(A_memtype == X_memtype, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Failed memtype guarantee in MatCreateDenseFromVecType");

  /* test */
  PetscCall(MatViewFromOptions(A, NULL, "-ex19_mat_view"));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&X));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: cuda
      requires: cuda
      args: -lda {{0 1}} -vec_type cuda -ex19_mat_view

   test:
      suffix: mpicuda
      requires: cuda
      args: -lda {{0 1}} -vec_type mpicuda -ex19_mat_view

   test:
      suffix: hip
      requires: hip
      args: -lda {{0 1}} -vec_type hip -ex19_mat_view

   test:
      suffix: standard
      args: -lda {{0 1}} -vec_type standard -ex19_mat_view

   test:
      suffix: kokkos_cuda
      requires: kokkos kokkos_kernels cuda
      args: -lda {{0 1}} -vec_type kokkos -ex19_mat_view

   test:
      suffix: kokkos_hip
      requires: kokkos kokkos_kernels hip
      args: -lda {{0 1}} -vec_type kokkos -ex19_mat_view

   test:
      suffix: kokkos
      requires: kokkos kokkos_kernels !cuda !hip
      args: -lda {{0 1}} -vec_type kokkos -ex19_mat_view
TEST*/
