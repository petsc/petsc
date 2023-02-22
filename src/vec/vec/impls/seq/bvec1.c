
/*
   Defines the BLAS based vector operations. Code shared by parallel
  and sequential vectors.
*/

#include <../src/vec/vec/impls/dvecimpl.h> /*I "petscvec.h" I*/
#include <petscblaslapack.h>

static PetscErrorCode VecXDot_Seq_Private(Vec xin, Vec yin, PetscScalar *z, PetscScalar (*const BLASfn)(const PetscBLASInt *, const PetscScalar *, const PetscBLASInt *, const PetscScalar *, const PetscBLASInt *))
{
  const PetscInt     n   = xin->map->n;
  const PetscBLASInt one = 1;
  const PetscScalar *ya, *xa;
  PetscBLASInt       bn;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(n, &bn));
  if (n > 0) PetscCall(PetscLogFlops(2.0 * n - 1));
  PetscCall(VecGetArrayRead(xin, &xa));
  PetscCall(VecGetArrayRead(yin, &ya));
  /* arguments ya, xa are reversed because BLAS complex conjugates the first argument, PETSc
     the second */
  PetscCallBLAS("BLASdot", *z = BLASfn(&bn, ya, &one, xa, &one));
  PetscCall(VecRestoreArrayRead(xin, &xa));
  PetscCall(VecRestoreArrayRead(yin, &ya));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecDot_Seq(Vec xin, Vec yin, PetscScalar *z)
{
  PetscFunctionBegin;
  PetscCall(VecXDot_Seq_Private(xin, yin, z, BLASdot_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecTDot_Seq(Vec xin, Vec yin, PetscScalar *z)
{
  PetscFunctionBegin;
  /*
    pay close attention!!! xin and yin are SWAPPED here so that the eventual BLAS call is
    dot(&bn, xa, &one, ya, &one)
  */
  PetscCall(VecXDot_Seq_Private(yin, xin, z, BLASdotu_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecScale_Seq(Vec xin, PetscScalar alpha)
{
  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    PetscCall(VecSet_Seq(xin, alpha));
  } else if (alpha != (PetscScalar)1.0) {
    const PetscBLASInt one = 1;
    PetscBLASInt       bn;
    PetscScalar       *xarray;

    PetscCall(PetscBLASIntCast(xin->map->n, &bn));
    PetscCall(PetscLogFlops(bn));
    PetscCall(VecGetArray(xin, &xarray));
    PetscCallBLAS("BLASscal", BLASscal_(&bn, &alpha, xarray, &one));
    PetscCall(VecRestoreArray(xin, &xarray));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecAXPY_Seq(Vec yin, PetscScalar alpha, Vec xin)
{
  PetscFunctionBegin;
  /* assume that the BLAS handles alpha == 1.0 efficiently since we have no fast code for it */
  if (alpha != (PetscScalar)0.0) {
    const PetscScalar *xarray;
    PetscScalar       *yarray;
    const PetscBLASInt one = 1;
    PetscBLASInt       bn;

    PetscCall(PetscBLASIntCast(yin->map->n, &bn));
    PetscCall(PetscLogFlops(2.0 * bn));
    PetscCall(VecGetArrayRead(xin, &xarray));
    PetscCall(VecGetArray(yin, &yarray));
    PetscCallBLAS("BLASaxpy", BLASaxpy_(&bn, &alpha, xarray, &one, yarray, &one));
    PetscCall(VecRestoreArrayRead(xin, &xarray));
    PetscCall(VecRestoreArray(yin, &yarray));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecAXPBY_Seq(Vec yin, PetscScalar a, PetscScalar b, Vec xin)
{
  PetscFunctionBegin;
  if (a == (PetscScalar)0.0) {
    PetscCall(VecScale_Seq(yin, b));
  } else if (b == (PetscScalar)1.0) {
    PetscCall(VecAXPY_Seq(yin, a, xin));
  } else if (a == (PetscScalar)1.0) {
    PetscCall(VecAYPX_Seq(yin, b, xin));
  } else {
    const PetscInt     n = yin->map->n;
    const PetscScalar *xx;
    PetscInt           flops;
    PetscScalar       *yy;

    PetscCall(VecGetArrayRead(xin, &xx));
    PetscCall(VecGetArray(yin, &yy));
    if (b == (PetscScalar)0.0) {
      flops = n;
      for (PetscInt i = 0; i < n; ++i) yy[i] = a * xx[i];
    } else {
      flops = 3 * n;
      for (PetscInt i = 0; i < n; ++i) yy[i] = a * xx[i] + b * yy[i];
    }
    PetscCall(VecRestoreArrayRead(xin, &xx));
    PetscCall(VecRestoreArray(yin, &yy));
    PetscCall(PetscLogFlops(flops));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecAXPBYPCZ_Seq(Vec zin, PetscScalar alpha, PetscScalar beta, PetscScalar gamma, Vec xin, Vec yin)
{
  const PetscInt     n = zin->map->n;
  const PetscScalar *yy, *xx;
  PetscInt           flops = 4 * n; // common case
  PetscScalar       *zz;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin, &xx));
  PetscCall(VecGetArrayRead(yin, &yy));
  PetscCall(VecGetArray(zin, &zz));
  if (alpha == (PetscScalar)1.0) {
    for (PetscInt i = 0; i < n; ++i) zz[i] = xx[i] + beta * yy[i] + gamma * zz[i];
  } else if (gamma == (PetscScalar)1.0) {
    for (PetscInt i = 0; i < n; ++i) zz[i] = alpha * xx[i] + beta * yy[i] + zz[i];
  } else if (gamma == (PetscScalar)0.0) {
    for (PetscInt i = 0; i < n; ++i) zz[i] = alpha * xx[i] + beta * yy[i];
    flops -= n;
  } else {
    for (PetscInt i = 0; i < n; ++i) zz[i] = alpha * xx[i] + beta * yy[i] + gamma * zz[i];
    flops += n;
  }
  PetscCall(VecRestoreArrayRead(xin, &xx));
  PetscCall(VecRestoreArrayRead(yin, &yy));
  PetscCall(VecRestoreArray(zin, &zz));
  PetscCall(PetscLogFlops(flops));
  PetscFunctionReturn(PETSC_SUCCESS);
}
