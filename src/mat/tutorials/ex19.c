const char help[] = "Test VecCreateMatDense()\n\n";

#include <petscdevice_cuda.h>
#include <petscmat.h>
#include <petscconf.h>
#include <assert.h>

int main(int argc, char **args)
{
  Mat      A;
  Vec      X;
  PetscInt N = 20;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Creating Mat from Vec example", NULL);
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &N, NULL));
  PetscOptionsEnd();

  PetscCall(VecCreate(PETSC_COMM_WORLD, &X));
  PetscCall(VecSetSizes(X, PETSC_DECIDE, N));
  PetscCall(VecSetFromOptions(X));
  PetscCall(VecSetUp(X));

  PetscCall(VecCreateMatDense(X, PETSC_DECIDE, PETSC_DECIDE, N, N, NULL, &A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  MPI_Comm    X_comm = PetscObjectComm((PetscObject)X);
  MPI_Comm    A_comm = PetscObjectComm((PetscObject)X);
  PetscMPIInt comp;
  PetscCallMPI(MPI_Comm_compare(X_comm, A_comm, &comp));
  PetscAssert(comp == MPI_IDENT || comp == MPI_CONGRUENT, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Failed communicator guarantee in MatCreateDenseMatchingVec()");

  PetscMemType       X_memtype, A_memtype;
  const PetscScalar *array;
  PetscCall(VecGetArrayReadAndMemType(X, &array, &X_memtype));
  PetscCall(VecRestoreArrayReadAndMemType(X, &array));
  PetscCall(MatDenseGetArrayReadAndMemType(A, &array, &A_memtype));
  PetscCall(MatDenseRestoreArrayReadAndMemType(A, &array));
  PetscAssert(A_memtype == X_memtype, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Failed memtype guarantee in MatCreateDenseMatchingVec()");

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
      args: -vec_type cuda -ex19_mat_view

   test:
      suffix: mpicuda
      requires: cuda
      args: -vec_type mpicuda -ex19_mat_view

   test:
      suffix: hip
      requires: hip
      args: -vec_type hip -ex19_mat_view

   test:
      suffix: standard
      args: -vec_type standard -ex19_mat_view

   test:
      suffix: kokkos_cuda
      requires: kokkos kokkos_kernels cuda
      args: -vec_type kokkos -ex19_mat_view

   test:
      suffix: kokkos_hip
      requires: kokkos kokkos_kernels hip
      args: -vec_type kokkos -ex19_mat_view

   test:
      suffix: kokkos
      requires: kokkos kokkos_kernels !cuda !hip
      args: -vec_type kokkos -ex19_mat_view
TEST*/
