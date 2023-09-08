static const char help[] = "Test PetscSFFetchAndOp \n\n";

#include <petscvec.h>
#include <petscsf.h>

int main(int argc, char *argv[])
{
  PetscInt           n, N = 12;
  PetscInt          *indices;
  IS                 ix, iy;
  VecScatter         vscat;
  Vec                x, y, z;
  PetscInt           rstart, rend;
  const PetscScalar *xarray;
  PetscScalar       *yarray, *zarray;
  PetscMemType       xmtype, ymtype, zmtype;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(VecCreateFromOptions(PETSC_COMM_WORLD, NULL, 1, PETSC_DECIDE, N, &x));
  PetscCall(VecDuplicate(x, &y));
  PetscCall(VecDuplicate(x, &z));
  PetscCall(VecGetLocalSize(x, &n));

  PetscCall(VecGetOwnershipRange(x, &rstart, &rend));
  PetscCall(ISCreateStride(PETSC_COMM_WORLD, n, rstart, 1, &ix));
  PetscCall(PetscMalloc1(n, &indices));
  for (PetscInt i = rstart; i < rend; i++) indices[i - rstart] = i / 4;
  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, n, indices, PETSC_OWN_POINTER, &iy));

  // connect y[0] to x[0..3], y[1] to x[4..7], etc
  PetscCall(VecScatterCreate(y, iy, x, ix, &vscat)); // y has roots, x has leaves

  PetscCall(VecSet(x, 1.0));
  PetscCall(VecSet(y, 2.0));

  PetscCall(VecGetArrayReadAndMemType(x, &xarray, &xmtype));
  PetscCall(VecGetArrayAndMemType(y, &yarray, &ymtype));
  PetscCall(VecGetArrayWriteAndMemType(z, &zarray, &zmtype));

  PetscCall(PetscSFFetchAndOpWithMemTypeBegin(vscat, MPIU_SCALAR, ymtype, yarray, xmtype, xarray, zmtype, zarray, MPI_SUM));
  PetscCall(PetscSFFetchAndOpEnd(vscat, MPIU_SCALAR, yarray, xarray, zarray, MPI_SUM));

  PetscCall(VecRestoreArrayReadAndMemType(x, &xarray));
  PetscCall(VecRestoreArrayAndMemType(y, &yarray));
  PetscCall(VecRestoreArrayWriteAndMemType(z, &zarray));

  PetscCall(VecView(y, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(z, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISDestroy(&ix));
  PetscCall(ISDestroy(&iy));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&z));
  PetscCall(VecScatterDestroy(&vscat));
  PetscCall(PetscFinalize());
}

/*TEST
  testset:
    nsize: {{1 4}}
    # since FetchAndOp on complex would need to be atomic in this test
    requires: !complex
    output_file: output/ex22.out
    filter: grep -v "type" | grep -v "Process" |grep -v "Vec Object"

    test:
      suffix: cuda
      requires: cuda
      args: -vec_type cuda

    test:
      suffix: hip
      requires: hip
      args: -vec_type hip

    test:
      suffix: kok
      requires: kokkos_kernels
      args: -vec_type kokkos

TEST*/
