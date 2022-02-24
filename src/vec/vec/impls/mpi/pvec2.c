
/*
     Code for some of the parallel vector primatives.
*/
#include <../src/vec/vec/impls/mpi/pvecimpl.h>
#include <petscblaslapack.h>

PetscErrorCode VecMDot_MPI(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;

  PetscFunctionBegin;
  if (nv > 128) {
    CHKERRQ(PetscMalloc1(nv,&work));
  }
  CHKERRQ(VecMDot_Seq(xin,nv,y,work));
  CHKERRMPI(MPIU_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  if (nv > 128) {
    CHKERRQ(PetscFree(work));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecMTDot_MPI(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;

  PetscFunctionBegin;
  if (nv > 128) {
    CHKERRQ(PetscMalloc1(nv,&work));
  }
  CHKERRQ(VecMTDot_Seq(xin,nv,y,work));
  CHKERRMPI(MPIU_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  if (nv > 128) {
    CHKERRQ(PetscFree(work));
  }
  PetscFunctionReturn(0);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/fnorm.h>
PetscErrorCode VecNorm_MPI(Vec xin,NormType type,PetscReal *z)
{
  PetscReal         sum,work = 0.0;
  const PetscScalar *xx;
  PetscInt          n   = xin->map->n;
  PetscBLASInt      one = 1,bn = 0;

  PetscFunctionBegin;
  CHKERRQ(PetscBLASIntCast(n,&bn));
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    CHKERRQ(VecGetArrayRead(xin,&xx));
    work = PetscRealPart(BLASdot_(&bn,xx,&one,xx,&one));
    CHKERRQ(VecRestoreArrayRead(xin,&xx));
    CHKERRMPI(MPIU_Allreduce(&work,&sum,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
    *z   = PetscSqrtReal(sum);
    CHKERRQ(PetscLogFlops(2.0*xin->map->n));
  } else if (type == NORM_1) {
    /* Find the local part */
    CHKERRQ(VecNorm_Seq(xin,NORM_1,&work));
    /* Find the global max */
    CHKERRMPI(MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
  } else if (type == NORM_INFINITY) {
    /* Find the local max */
    CHKERRQ(VecNorm_Seq(xin,NORM_INFINITY,&work));
    /* Find the global max */
    CHKERRMPI(MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin)));
  } else if (type == NORM_1_AND_2) {
    PetscReal temp[2];
    CHKERRQ(VecNorm_Seq(xin,NORM_1,temp));
    CHKERRQ(VecNorm_Seq(xin,NORM_2,temp+1));
    temp[1] = temp[1]*temp[1];
    CHKERRMPI(MPIU_Allreduce(temp,z,2,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin)));
    z[1] = PetscSqrtReal(z[1]);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecMax_MPI(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscReal      work;

  PetscFunctionBegin;
  /* Find the local max */
  CHKERRQ(VecMax_Seq(xin,idx,&work));
#if defined(PETSC_HAVE_MPIUNI)
  *z = work;
#else
  /* Find the global max */
  if (!idx) {
    CHKERRMPI(MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin)));
  } else {
    struct { PetscReal v; PetscInt i; } in,out;

    in.v  = work;
    in.i  = *idx + xin->map->rstart;
    CHKERRMPI(MPIU_Allreduce(&in,&out,1,MPIU_REAL_INT,MPIU_MAXLOC,PetscObjectComm((PetscObject)xin)));
    *z    = out.v;
    *idx  = out.i;
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode VecMin_MPI(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscReal      work;

  PetscFunctionBegin;
  /* Find the local Min */
  CHKERRQ(VecMin_Seq(xin,idx,&work));
#if defined(PETSC_HAVE_MPIUNI)
  *z = work;
#else
  /* Find the global Min */
  if (!idx) {
    CHKERRMPI(MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)xin)));
  } else {
    struct { PetscReal v; PetscInt i; } in,out;

    in.v  = work;
    in.i  = *idx + xin->map->rstart;
    CHKERRMPI(MPIU_Allreduce(&in,&out,1,MPIU_REAL_INT,MPIU_MINLOC,PetscObjectComm((PetscObject)xin)));
    *z    = out.v;
    *idx  = out.i;
  }
#endif
  PetscFunctionReturn(0);
}
