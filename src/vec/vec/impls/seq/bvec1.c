
/*
   Defines the BLAS based vector operations. Code shared by parallel
  and sequential vectors.
*/

#include <../src/vec/vec/impls/dvecimpl.h>          /*I "petscvec.h" I*/
#include <petscblaslapack.h>

PetscErrorCode VecDot_Seq(Vec xin,Vec yin,PetscScalar *z)
{
  const PetscScalar *ya,*xa;
  PetscBLASInt      one = 1,bn = 0;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(xin->map->n,&bn));
  PetscCall(VecGetArrayRead(xin,&xa));
  PetscCall(VecGetArrayRead(yin,&ya));
  /* arguments ya, xa are reversed because BLAS complex conjugates the first argument, PETSc the second */
  PetscStackCallBLAS("BLASdot",*z   = BLASdot_(&bn,ya,&one,xa,&one));
  PetscCall(VecRestoreArrayRead(xin,&xa));
  PetscCall(VecRestoreArrayRead(yin,&ya));
  if (xin->map->n > 0) {
    PetscCall(PetscLogFlops(2.0*xin->map->n-1));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_Seq(Vec xin,Vec yin,PetscScalar *z)
{
  const PetscScalar *ya,*xa;
  PetscBLASInt      one = 1,bn = 0;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(xin->map->n,&bn));
  PetscCall(VecGetArrayRead(xin,&xa));
  PetscCall(VecGetArrayRead(yin,&ya));
  PetscStackCallBLAS("BLASdot",*z   = BLASdotu_(&bn,xa,&one,ya,&one));
  PetscCall(VecRestoreArrayRead(xin,&xa));
  PetscCall(VecRestoreArrayRead(yin,&ya));
  if (xin->map->n > 0) {
    PetscCall(PetscLogFlops(2.0*xin->map->n-1));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecScale_Seq(Vec xin, PetscScalar alpha)
{
  PetscBLASInt   one = 1,bn;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(xin->map->n,&bn));
  if (alpha == (PetscScalar)0.0) {
    PetscCall(VecSet_Seq(xin,alpha));
  } else if (alpha != (PetscScalar)1.0) {
    PetscScalar a = alpha,*xarray;
    PetscCall(VecGetArray(xin,&xarray));
    PetscStackCallBLAS("BLASscal",BLASscal_(&bn,&a,xarray,&one));
    PetscCall(VecRestoreArray(xin,&xarray));
  }
  PetscCall(PetscLogFlops(xin->map->n));
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPY_Seq(Vec yin,PetscScalar alpha,Vec xin)
{
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscBLASInt      one = 1,bn;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(yin->map->n,&bn));
  /* assume that the BLAS handles alpha == 1.0 efficiently since we have no fast code for it */
  if (alpha != (PetscScalar)0.0) {
    PetscCall(VecGetArrayRead(xin,&xarray));
    PetscCall(VecGetArray(yin,&yarray));
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&bn,&alpha,xarray,&one,yarray,&one));
    PetscCall(VecRestoreArrayRead(xin,&xarray));
    PetscCall(VecRestoreArray(yin,&yarray));
    PetscCall(PetscLogFlops(2.0*yin->map->n));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBY_Seq(Vec yin,PetscScalar a,PetscScalar b,Vec xin)
{
  PetscInt          n = yin->map->n,i;
  const PetscScalar *xx;
  PetscScalar       *yy;

  PetscFunctionBegin;
  if (a == (PetscScalar)0.0) {
    PetscCall(VecScale_Seq(yin,b));
  } else if (b == (PetscScalar)1.0) {
    PetscCall(VecAXPY_Seq(yin,a,xin));
  } else if (a == (PetscScalar)1.0) {
    PetscCall(VecAYPX_Seq(yin,b,xin));
  } else if (b == (PetscScalar)0.0) {
    PetscCall(VecGetArrayRead(xin,&xx));
    PetscCall(VecGetArray(yin,(PetscScalar**)&yy));
    for (i=0; i<n; i++) yy[i] = a*xx[i];
    PetscCall(VecRestoreArrayRead(xin,&xx));
    PetscCall(VecRestoreArray(yin,(PetscScalar**)&yy));
    PetscCall(PetscLogFlops(xin->map->n));
  } else {
    PetscCall(VecGetArrayRead(xin,&xx));
    PetscCall(VecGetArray(yin,(PetscScalar**)&yy));
    for (i=0; i<n; i++) yy[i] = a*xx[i] + b*yy[i];
    PetscCall(VecRestoreArrayRead(xin,&xx));
    PetscCall(VecRestoreArray(yin,(PetscScalar**)&yy));
    PetscCall(PetscLogFlops(3.0*xin->map->n));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBYPCZ_Seq(Vec zin,PetscScalar alpha,PetscScalar beta,PetscScalar gamma,Vec xin,Vec yin)
{
  PetscInt          n = zin->map->n,i;
  const PetscScalar *yy,*xx;
  PetscScalar       *zz;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin,&xx));
  PetscCall(VecGetArrayRead(yin,&yy));
  PetscCall(VecGetArray(zin,&zz));
  if (alpha == (PetscScalar)1.0) {
    for (i=0; i<n; i++) zz[i] = xx[i] + beta*yy[i] + gamma*zz[i];
    PetscCall(PetscLogFlops(4.0*n));
  } else if (gamma == (PetscScalar)1.0) {
    for (i=0; i<n; i++) zz[i] = alpha*xx[i] + beta*yy[i] + zz[i];
    PetscCall(PetscLogFlops(4.0*n));
  } else if (gamma == (PetscScalar)0.0) {
    for (i=0; i<n; i++) zz[i] = alpha*xx[i] + beta*yy[i];
    PetscCall(PetscLogFlops(3.0*n));
  } else {
    for (i=0; i<n; i++) zz[i] = alpha*xx[i] + beta*yy[i] + gamma*zz[i];
    PetscCall(PetscLogFlops(5.0*n));
  }
  PetscCall(VecRestoreArrayRead(xin,&xx));
  PetscCall(VecRestoreArrayRead(yin,&yy));
  PetscCall(VecRestoreArray(zin,&zz));
  PetscFunctionReturn(0);
}
