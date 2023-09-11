static const char help[] = "Test PetscSF with integers and MPIU_2INT \n\n";

#include <petscvec.h>
#include <petscsf.h>
#include <petscdevice.h>

int main(int argc, char *argv[])
{
  PetscInt           n, n2, N = 12;
  PetscInt          *indices;
  IS                 ix, iy;
  VecScatter         vscat;
  Vec                x, y;
  PetscInt           rstart, rend;
  PetscInt          *xh, *yh, *xd, *yd;
  PetscDeviceContext dctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(VecCreateFromOptions(PETSC_COMM_WORLD, NULL, 1, PETSC_DECIDE, N, &x));
  PetscCall(VecDuplicate(x, &y));
  PetscCall(VecGetLocalSize(x, &n));

  PetscCall(VecGetOwnershipRange(x, &rstart, &rend));
  PetscCall(ISCreateStride(PETSC_COMM_WORLD, n, rstart, 1, &ix));
  PetscCall(PetscMalloc1(n, &indices));
  for (int i = rstart; i < rend; i++) indices[i - rstart] = i / 2;
  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, n, indices, PETSC_OWN_POINTER, &iy));
  // connect y[0] to x[0..1], y[1] to x[2..3], etc
  PetscCall(VecScatterCreate(y, iy, x, ix, &vscat)); // y has roots, x has leaves

  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));

  // double the allocation since we will use MPIU_2INT later
  n2 = 2 * n;
  PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, n2, &xh));
  PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, n2, &yh));
  PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_DEVICE, n2, &xd));
  PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_DEVICE, n2, &yd));

  for (PetscInt i = 0; i < n; i++) {
    xh[i] = xh[i + n] = i + rstart;
    yh[i] = yh[i + n] = i + rstart;
  }
  PetscCall(PetscDeviceMemcpy(dctx, xd, xh, sizeof(PetscInt) * n2));
  PetscCall(PetscDeviceMemcpy(dctx, yd, yh, sizeof(PetscInt) * n2));

  PetscCall(PetscSFReduceWithMemTypeBegin(vscat, MPIU_INT, PETSC_MEMTYPE_DEVICE, xd, PETSC_MEMTYPE_DEVICE, yd, MPI_SUM));
  PetscCall(PetscSFReduceEnd(vscat, MPIU_INT, xd, yd, MPI_SUM));
  PetscCall(PetscDeviceMemcpy(dctx, yh, yd, sizeof(PetscInt) * n));
  PetscCall(PetscDeviceContextSynchronize(dctx)); // finish the async memcpy
  PetscCall(PetscIntView(n, yh, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscSFBcastWithMemTypeBegin(vscat, MPIU_2INT, PETSC_MEMTYPE_DEVICE, yd, PETSC_MEMTYPE_DEVICE, xd, MPI_MINLOC));
  PetscCall(PetscSFBcastEnd(vscat, MPIU_2INT, yd, xd, MPI_MINLOC));
  PetscCall(PetscDeviceMemcpy(dctx, xh, xd, sizeof(PetscInt) * n2));
  PetscCall(PetscDeviceContextSynchronize(dctx)); // finish the async memcpy
  PetscCall(PetscIntView(n2, xh, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscDeviceFree(dctx, xh));
  PetscCall(PetscDeviceFree(dctx, yh));
  PetscCall(PetscDeviceFree(dctx, xd));
  PetscCall(PetscDeviceFree(dctx, yd));
  PetscCall(ISDestroy(&ix));
  PetscCall(ISDestroy(&iy));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecScatterDestroy(&vscat));
  PetscCall(PetscFinalize());
}

/*TEST
  testset:
    output_file: output/ex23.out
    nsize: 3

    test:
      suffix: 1
      requires: cuda

    test:
      suffix: 2
      requires: hip

    test:
      suffix: 3
      requires: sycl

TEST*/
