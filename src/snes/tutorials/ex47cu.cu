static char help[] = "Solves -Laplacian u - exp(u) = 0,  0 < x < 1 using GPU\n\n";
/*
   Same as ex47.c except it also uses the GPU to evaluate the function
*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>

#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/tuple.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

extern PetscErrorCode ComputeFunction(SNES, Vec, Vec, void *), ComputeJacobian(SNES, Vec, Mat, Mat, void *);
PetscBool             useCUDA = PETSC_FALSE;

int main(int argc, char **argv)
{
  SNES      snes;
  Vec       x, f;
  Mat       J;
  DM        da;
  char     *tmp, typeName[256];
  PetscBool flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-dm_vec_type", typeName, sizeof(typeName), &flg));
  if (flg) {
    PetscCall(PetscStrstr(typeName, "cuda", &tmp));
    if (tmp) useCUDA = PETSC_TRUE;
  }

  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 8, 1, 1, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMCreateGlobalVector(da, &x));
  PetscCall(VecDuplicate(x, &f));
  PetscCall(DMCreateMatrix(da, &J));

  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetFunction(snes, f, ComputeFunction, da));
  PetscCall(SNESSetJacobian(snes, J, J, ComputeJacobian, da));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(SNESSolve(snes, NULL, x));

  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&f));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&da));

  PetscCall(PetscFinalize());
  return 0;
}

struct ApplyStencil {
  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t)
  {
    /* f = (2*x_i - x_(i+1) - x_(i-1))/h - h*exp(x_i) */
    thrust::get<0>(t) = 1;
    if ((thrust::get<4>(t) > 0) && (thrust::get<4>(t) < thrust::get<5>(t) - 1)) {
      thrust::get<0>(t) = (((PetscScalar)2.0) * thrust::get<1>(t) - thrust::get<2>(t) - thrust::get<3>(t)) / (thrust::get<6>(t)) - (thrust::get<6>(t)) * exp(thrust::get<1>(t));
    } else if (thrust::get<4>(t) == 0) {
      thrust::get<0>(t) = thrust::get<1>(t) / (thrust::get<6>(t));
    } else if (thrust::get<4>(t) == thrust::get<5>(t) - 1) {
      thrust::get<0>(t) = thrust::get<1>(t) / (thrust::get<6>(t));
    }
  }
};

PetscErrorCode ComputeFunction(SNES snes, Vec x, Vec f, void *ctx)
{
  PetscInt           i, Mx, xs, xm, xstartshift, xendshift, fstart, lsize;
  PetscScalar       *xx, *ff, hx;
  DM                 da = (DM)ctx;
  Vec                xlocal;
  PetscMPIInt        rank, size;
  MPI_Comm           comm;
  PetscScalar const *xarray;
  PetscScalar       *farray;

  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));
  hx = 1.0 / (PetscReal)(Mx - 1);
  PetscCall(DMGetLocalVector(da, &xlocal));
  PetscCall(DMGlobalToLocalBegin(da, x, INSERT_VALUES, xlocal));
  PetscCall(DMGlobalToLocalEnd(da, x, INSERT_VALUES, xlocal));

  if (useCUDA) {
    PetscCall(PetscObjectGetComm((PetscObject)da, &comm));
    PetscCallMPI(MPI_Comm_size(comm, &size));
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCall(VecCUDAGetArrayRead(xlocal, &xarray));
    PetscCall(VecCUDAGetArrayWrite(f, &farray));
    if (rank) xstartshift = 1;
    else xstartshift = 0;
    if (rank != size - 1) xendshift = 1;
    else xendshift = 0;
    PetscCall(VecGetOwnershipRange(f, &fstart, NULL));
    PetscCall(VecGetLocalSize(x, &lsize));
    // clang-format off
    try {
      thrust::for_each(
        thrust::make_zip_iterator(
          thrust::make_tuple(
            thrust::device_ptr<PetscScalar>(farray),
            thrust::device_ptr<const PetscScalar>(xarray + xstartshift),
            thrust::device_ptr<const PetscScalar>(xarray + xstartshift + 1),
            thrust::device_ptr<const PetscScalar>(xarray + xstartshift - 1),
            thrust::counting_iterator<int>(fstart),
            thrust::constant_iterator<int>(Mx),
            thrust::constant_iterator<PetscScalar>(hx))),
        thrust::make_zip_iterator(
          thrust::make_tuple(
            thrust::device_ptr<PetscScalar>(farray + lsize),
            thrust::device_ptr<const PetscScalar>(xarray + lsize - xendshift),
            thrust::device_ptr<const PetscScalar>(xarray + lsize - xendshift + 1),
            thrust::device_ptr<const PetscScalar>(xarray + lsize - xendshift - 1),
            thrust::counting_iterator<int>(fstart) + lsize,
            thrust::constant_iterator<int>(Mx),
            thrust::constant_iterator<PetscScalar>(hx))),
        ApplyStencil());
    }
    // clang-format on
    catch (char *all) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Thrust is not working\n"));
    }
    PetscCall(VecCUDARestoreArrayRead(xlocal, &xarray));
    PetscCall(VecCUDARestoreArrayWrite(f, &farray));
  } else {
    PetscCall(DMDAVecGetArray(da, xlocal, &xx));
    PetscCall(DMDAVecGetArray(da, f, &ff));
    PetscCall(DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL));

    for (i = xs; i < xs + xm; i++) {
      if (i == 0 || i == Mx - 1) ff[i] = xx[i] / hx;
      else ff[i] = (2.0 * xx[i] - xx[i - 1] - xx[i + 1]) / hx - hx * PetscExpScalar(xx[i]);
    }
    PetscCall(DMDAVecRestoreArray(da, xlocal, &xx));
    PetscCall(DMDAVecRestoreArray(da, f, &ff));
  }
  PetscCall(DMRestoreLocalVector(da, &xlocal));
  return 0;
}
PetscErrorCode ComputeJacobian(SNES snes, Vec x, Mat J, Mat B, void *ctx)
{
  DM          da = (DM)ctx;
  PetscInt    i, Mx, xm, xs;
  PetscScalar hx, *xx;
  Vec         xlocal;

  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));
  hx = 1.0 / (PetscReal)(Mx - 1);
  PetscCall(DMGetLocalVector(da, &xlocal));
  PetscCall(DMGlobalToLocalBegin(da, x, INSERT_VALUES, xlocal));
  PetscCall(DMGlobalToLocalEnd(da, x, INSERT_VALUES, xlocal));
  PetscCall(DMDAVecGetArray(da, xlocal, &xx));
  PetscCall(DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL));

  for (i = xs; i < xs + xm; i++) {
    if (i == 0 || i == Mx - 1) {
      PetscCall(MatSetValue(J, i, i, 1.0 / hx, INSERT_VALUES));
    } else {
      PetscCall(MatSetValue(J, i, i - 1, -1.0 / hx, INSERT_VALUES));
      PetscCall(MatSetValue(J, i, i, 2.0 / hx - hx * PetscExpScalar(xx[i]), INSERT_VALUES));
      PetscCall(MatSetValue(J, i, i + 1, -1.0 / hx, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  PetscCall(DMDAVecRestoreArray(da, xlocal, &xx));
  PetscCall(DMRestoreLocalVector(da, &xlocal));
  return 0;
}

/*TEST

   build:
      requires: cuda

   testset:
      args: -snes_monitor_short -dm_mat_type aijcusparse -dm_vec_type cuda
      output_file: output/ex47cu_1.out
      test:
        suffix: 1
        nsize:  1
      test:
        suffix: 2
        nsize:  2

TEST*/
