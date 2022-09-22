static const char help[] = "Test VecGetSubVector()\n\n";

#include <petscvec.h>

int main(int argc, char *argv[])
{
  MPI_Comm     comm;
  Vec          X, Y, Z, W;
  PetscMPIInt  rank, size;
  PetscInt     i, rstart, rend, idxs[3];
  PetscScalar *x, *y, *w, *z;
  PetscViewer  viewer;
  IS           is0, is1, is2;
  PetscBool    iscuda;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, 0, help));
  comm   = PETSC_COMM_WORLD;
  viewer = PETSC_VIEWER_STDOUT_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(VecCreate(comm, &X));
  PetscCall(VecSetSizes(X, 10, PETSC_DETERMINE));
  PetscCall(VecSetFromOptions(X));
  PetscCall(VecGetOwnershipRange(X, &rstart, &rend));

  PetscCall(VecGetArray(X, &x));
  for (i = 0; i < rend - rstart; i++) x[i] = rstart + i;
  PetscCall(VecRestoreArray(X, &x));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)X, &iscuda, VECSEQCUDA, VECMPICUDA, ""));
  if (iscuda) { /* trigger a copy of the data on the GPU */
    const PetscScalar *xx;

    PetscCall(VecCUDAGetArrayRead(X, &xx));
    PetscCall(VecCUDARestoreArrayRead(X, &xx));
  }

  PetscCall(VecView(X, viewer));

  idxs[0] = (size - rank - 1) * 10 + 5;
  idxs[1] = (size - rank - 1) * 10 + 2;
  idxs[2] = (size - rank - 1) * 10 + 3;

  PetscCall(ISCreateStride(comm, (rend - rstart) / 3 + 3 * (rank > size / 2), rstart, 1, &is0));
  PetscCall(ISComplement(is0, rstart, rend, &is1));
  PetscCall(ISCreateGeneral(comm, 3, idxs, PETSC_USE_POINTER, &is2));

  PetscCall(ISView(is0, viewer));
  PetscCall(ISView(is1, viewer));
  PetscCall(ISView(is2, viewer));

  PetscCall(VecGetSubVector(X, is0, &Y));
  PetscCall(VecGetSubVector(X, is1, &Z));
  PetscCall(VecGetSubVector(X, is2, &W));
  PetscCall(VecView(Y, viewer));
  PetscCall(VecView(Z, viewer));
  PetscCall(VecView(W, viewer));
  PetscCall(VecGetArray(Y, &y));
  y[0] = 1000 * (rank + 1);
  PetscCall(VecRestoreArray(Y, &y));
  PetscCall(VecGetArray(Z, &z));
  z[0] = -1000 * (rank + 1);
  PetscCall(VecRestoreArray(Z, &z));
  PetscCall(VecGetArray(W, &w));
  w[0] = -10 * (rank + 1);
  PetscCall(VecRestoreArray(W, &w));
  PetscCall(VecRestoreSubVector(X, is0, &Y));
  PetscCall(VecRestoreSubVector(X, is1, &Z));
  PetscCall(VecRestoreSubVector(X, is2, &W));
  PetscCall(VecView(X, viewer));

  PetscCall(ISDestroy(&is0));
  PetscCall(ISDestroy(&is1));
  PetscCall(ISDestroy(&is2));
  PetscCall(VecDestroy(&X));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      nsize: 3
      output_file: output/ex38_1.out
      filter: grep -v "  type:"
      diff_args: -j
      test:
        suffix: standard
        args: -vec_type standard
      test:
        requires: cuda
        suffix: cuda
        args: -vec_type cuda
      test:
        requires: viennacl
        suffix:  viennacl
        args: -vec_type viennacl
      test:
        requires: kokkos_kernels
        suffix: kokkos
        args: -vec_type kokkos
      test:
        requires: hip
        suffix: hip
        args: -vec_type hip

TEST*/
