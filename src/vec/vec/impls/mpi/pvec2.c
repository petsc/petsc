
/*
     Code for some of the parallel vector primatives.
*/
#include <../src/vec/vec/impls/mpi/pvecimpl.h>
#include <petscblaslapack.h>

PetscErrorCode VecMDot_MPI(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nv > 128) {
    ierr = PetscMalloc1(nv,&work);CHKERRQ(ierr);
  }
  ierr = VecMDot_Seq(xin,nv,y,work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  if (nv > 128) {
    ierr = PetscFree(work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecMTDot_MPI(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nv > 128) {
    ierr = PetscMalloc1(nv,&work);CHKERRQ(ierr);
  }
  ierr = VecMTDot_Seq(xin,nv,y,work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  if (nv > 128) {
    ierr = PetscFree(work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/fnorm.h>
PetscErrorCode VecNorm_MPI(Vec xin,NormType type,PetscReal *z)
{
  PetscReal         sum,work = 0.0;
  const PetscScalar *xx;
  PetscErrorCode    ierr;
  PetscInt          n   = xin->map->n;
  PetscBLASInt      one = 1,bn = 0;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
    work = PetscRealPart(BLASdot_(&bn,xx,&one,xx,&one));
    ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(&work,&sum,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
    *z   = PetscSqrtReal(sum);
    ierr = PetscLogFlops(2.0*xin->map->n);CHKERRQ(ierr);
  } else if (type == NORM_1) {
    /* Find the local part */
    ierr = VecNorm_Seq(xin,NORM_1,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  } else if (type == NORM_INFINITY) {
    /* Find the local max */
    ierr = VecNorm_Seq(xin,NORM_INFINITY,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  } else if (type == NORM_1_AND_2) {
    PetscReal temp[2];
    ierr = VecNorm_Seq(xin,NORM_1,temp);CHKERRQ(ierr);
    ierr = VecNorm_Seq(xin,NORM_2,temp+1);CHKERRQ(ierr);
    temp[1] = temp[1]*temp[1];
    ierr = MPIU_Allreduce(temp,z,2,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
    z[1] = PetscSqrtReal(z[1]);
  }
  PetscFunctionReturn(0);
}

extern MPI_Op MPIU_MAXINDEX_OP, MPIU_MININDEX_OP;

PetscErrorCode VecMax_MPI(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscErrorCode ierr;
  PetscReal      work;

  PetscFunctionBegin;
  /* Find the local max */
  ierr = VecMax_Seq(xin,idx,&work);CHKERRQ(ierr);

  /* Find the global max */
  if (!idx) {
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  } else {
    PetscReal work2[2],z2[2];
    PetscInt  rstart;
    rstart   = xin->map->rstart;
    work2[0] = work;
    work2[1] = *idx + rstart;
    ierr     = MPIU_Allreduce(work2,z2,2,MPIU_REAL,MPIU_MAXINDEX_OP,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
    *z       = z2[0];
    *idx     = (PetscInt)z2[1];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecMin_MPI(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscErrorCode ierr;
  PetscReal      work;

  PetscFunctionBegin;
  /* Find the local Min */
  ierr = VecMin_Seq(xin,idx,&work);CHKERRQ(ierr);

  /* Find the global Min */
  if (!idx) {
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
  } else {
    PetscReal work2[2],z2[2];
    PetscInt  rstart;

    ierr = VecGetOwnershipRange(xin,&rstart,NULL);CHKERRQ(ierr);
    work2[0] = work;
    work2[1] = *idx + rstart;
    ierr = MPIU_Allreduce(work2,z2,2,MPIU_REAL,MPIU_MININDEX_OP,PetscObjectComm((PetscObject)xin));CHKERRMPI(ierr);
    *z   = z2[0];
    *idx = (PetscInt)z2[1];
  }
  PetscFunctionReturn(0);
}
